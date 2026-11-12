import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantic_spotter import gitsource
from semantic_spotter.embedder import MAX_BATCH_TOKENS, Embedder, batches
from semantic_spotter.extractor import extract, snippet_text
from semantic_spotter.filters import IGNORE_FILENAME, PathFilter, parse_ignore_file
from semantic_spotter.languages import spec_for
from semantic_spotter.models import Snippet, TreeFile
from semantic_spotter.store import ExtractedBlob, Sighting, Store


# Sube cuando cambia lo que se extrae por snippet, para forzar la re-extracción.
EXTRACT_VERSION = 4


@dataclass(frozen=True)
class ExtractOptions:
    min_func_lines: int = 3
    min_block_lines: int = 6
    include_tests: bool = False

    @property
    def key(self) -> str:
        """Identifica la configuración con la que se extrajo un blob.

        `include_tests` no forma parte: elige qué archivos se miran, no cómo se extraen.
        """
        return f"v{EXTRACT_VERSION}-f{self.min_func_lines}-b{self.min_block_lines}"

    @property
    def normalizer(self) -> str:
        """Versión de la huella estructural (`norm_hash`, shingles) por contenido."""
        return f"v{EXTRACT_VERSION}"


def _extract_blob(file: TreeFile, source: bytes, options: ExtractOptions) -> ExtractedBlob:
    spec = spec_for(file.path)
    generated = gitsource.is_generated(source)
    snippets = []
    if spec is not None and not generated:
        snippets = extract(
            spec,
            file.path,
            source,
            file.blob_oid,
            min_func_lines=options.min_func_lines,
            min_block_lines=options.min_block_lines,
        )
    return ExtractedBlob(file.blob_oid, file.language, generated, snippets)


async def tree_files(store: Store, repo: Path, tree_oid: str) -> list[TreeFile]:
    """Archivos de lenguajes soportados del árbol; se registran la primera vez."""
    if await store.has_tree(tree_oid):
        return await store.load_tree(tree_oid)
    files = [
        TreeFile(f.path, f.oid, spec.name)
        for f in await gitsource.list_files(repo, tree_oid)
        if (spec := spec_for(f.path)) and gitsource.is_candidate(f.path, include_tests=True)
    ]
    async with store.writing():
        await store.save_tree(tree_oid, files)
    return files


async def index_tree(
    store: Store, repo: Path, tree_oid: str, options: ExtractOptions
) -> list[Snippet]:
    """Extrae solo los blobs del árbol que aún no están en la DB y devuelve sus snippets."""
    files = [
        f
        for f in await tree_files(store, repo, tree_oid)
        if gitsource.is_candidate(f.path, include_tests=options.include_tests)
    ]
    keys = [(f.blob_oid, f.language) for f in files]
    if await store.missing_blobs(keys, options.key):
        async with store.writing():
            # Otro proceso pudo haber guardado algunos mientras esperábamos el lock.
            missing = await store.missing_blobs(keys, options.key)
            file_of = {(f.blob_oid, f.language): f for f in files}
            sources = await gitsource.read_blobs(repo, sorted({oid for oid, _ in missing}))
            # El parseo con tree-sitter es CPU pura y rápida: se hace en línea, sin threads.
            extracted = [
                _extract_blob(file_of[k], sources[k[0]], options)
                for k in sorted(missing)
                if k[0] in sources
            ]
            await store.save_blobs(options.key, options.normalizer, extracted)
    return await store.load_snippets(files, options.key, options.normalizer)


async def index_commit(
    store: Store, repo: Path, sha: str, options: ExtractOptions
) -> list[Snippet]:
    """Registra que el commit se analizó y devuelve los snippets de su árbol."""
    tree_oid = await gitsource.tree_of(repo, sha)
    async with store.writing():
        await store.link_commit(sha, tree_oid)
    return await index_tree(store, repo, tree_oid, options)


# Commits por lote al recorrer el historial: cada lote lee sus blobs con un solo
# `git cat-file` y se guarda de una vez.
_HISTORY_BATCH = 200


@dataclass(frozen=True)
class HistoryStats:
    commits: int
    walked: int
    blobs: int
    sightings: int


async def index_history(
    store: Store,
    repo: Path,
    rev: str,
    since: str | None,
    options: ExtractOptions,
    on_batch: Callable[[int, int], None] | None = None,
) -> HistoryStats:
    """Recorre los commits y registra el primero en que apareció cada (ruta, contenido).

    Solo mira los archivos que cada commit cambió y no guarda sus snippets: los
    hashes se recalculan sin huella estructural, que es lo caro. Retoma desde
    donde quedó una corrida anterior con la misma config.
    """
    commits = await gitsource.history(repo, rev, since)
    walked = await store.walked_commits(options.key)
    todo = [c for c in commits if c.sha not in walked]
    seen = await store.seen_keys(options.key)
    hashes: dict[tuple[str, str], list[str]] = {}
    sightings_total = 0
    for start in range(0, len(todo), _HISTORY_BATCH):
        batch = todo[start : start + _HISTORY_BATCH]
        wanted = {
            (f.oid, spec.name): f
            for c in batch
            for f in c.changed
            if (spec := spec_for(f.path)) and gitsource.is_candidate(f.path, include_tests=True)
        }
        new = {k: f for k, f in wanted.items() if k not in hashes}
        sources = await gitsource.read_blobs(repo, sorted({oid for oid, _ in new}))
        for (oid, language), f in new.items():
            source = sources.get(oid)
            spec = spec_for(f.path)
            if source is None or spec is None or gitsource.is_generated(source):
                hashes[(oid, language)] = []
                continue
            snippets = extract(
                spec,
                f.path,
                source,
                oid,
                min_func_lines=options.min_func_lines,
                min_block_lines=options.min_block_lines,
                fingerprint=False,
            )
            hashes[(oid, language)] = [s.content_hash for s in snippets]
        sightings: list[Sighting] = []
        for c in batch:
            for f in c.changed:
                spec = spec_for(f.path)
                for h in hashes.get((f.oid, spec.name), []) if spec else []:
                    if (f.path, h) not in seen:
                        seen.add((f.path, h))
                        sightings.append(Sighting(f.path, h, c.sha, c.committed_at))
        async with store.writing():
            await store.save_history(
                options.key, [(c.sha, c.committed_at) for c in batch], sightings
            )
        sightings_total += len(sightings)
        if on_batch is not None:
            on_batch(start + len(batch), len(todo))
    return HistoryStats(len(commits), len(todo), len(hashes), sightings_total)


async def load_path_filter(
    repo: Path, tree_ish: str, extra_patterns: Sequence[str]
) -> PathFilter:
    """Patrones de `.semantic-spotterignore` en el árbol, seguidos de los de la CLI."""
    content = await gitsource.read_file(repo, tree_ish, IGNORE_FILENAME)
    patterns = parse_ignore_file(content.decode("utf-8", "replace")) if content else []
    return PathFilter([*patterns, *extra_patterns])


def apply_path_filter(snippets: Sequence[Snippet], path_filter: PathFilter) -> list[Snippet]:
    """Las exclusiones son un filtro de vista: no cambian lo extraído ni el caché."""
    if not path_filter:
        return list(snippets)
    return [s for s in snippets if not path_filter.excluded(s.file_path)]


@dataclass(frozen=True)
class EmbeddingPlan:
    hashes: list[str]
    texts: list[str]
    token_counts: list[int]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)


async def plan_embeddings(
    store: Store,
    repo: Path,
    snippets: Sequence[Snippet],
    embedder_factory: Callable[[], Embedder],
) -> EmbeddingPlan:
    """Hashes que todavía no tienen embedding en caché, con su texto leído de git.

    El embedder solo se construye si falta algo, así un re-escaneo sin cambios
    no necesita API key ni descarga el tokenizer.
    """
    occurrence: dict[str, Snippet] = {}
    for s in snippets:
        occurrence.setdefault(s.content_hash, s)
    missing = await store.missing_hashes(list(occurrence))
    if not missing:
        return EmbeddingPlan([], [], [])
    sources = await gitsource.read_blobs(repo, sorted({occurrence[h].blob_oid for h in missing}))
    texts = [
        snippet_text(sources[s.blob_oid], s.start_byte, s.end_byte)
        for s in (occurrence[h] for h in missing)
    ]
    return EmbeddingPlan(missing, texts, embedder_factory().token_counts(texts))


@dataclass(frozen=True)
class RateLimits:
    """Límites de la cuenta de Voyage; None = sin espaciar requests."""

    rpm: int | None = None
    tpm: int | None = None

    @property
    def batch_tokens(self) -> int:
        return min(MAX_BATCH_TOKENS, self.tpm or MAX_BATCH_TOKENS)

    def wait_after(self, tokens: int) -> float:
        """Segundos a esperar tras una request de `tokens` tokens."""
        waits = [0.0]
        if self.rpm:
            waits.append(60.0 / self.rpm)
        if self.tpm:
            waits.append(60.0 * tokens / self.tpm)
        return max(waits)


class RateLimiter:
    """Espacia el inicio de las requests concurrentes según `RateLimits`."""

    def __init__(
        self,
        limits: RateLimits,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits
        self._sleep = sleep
        self._clock = clock
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self, tokens: int) -> None:
        async with self._lock:
            if (delay := self._next_allowed - self._clock()) > 0:
                await self._sleep(delay)
            self._next_allowed = self._clock() + self._limits.wait_after(tokens)


async def run_embeddings(
    store: Store,
    embedder: Embedder,
    plan: EmbeddingPlan,
    on_batch: Callable[[int], None] | None = None,
    *,
    limits: RateLimits = RateLimits(),
    concurrency: int = 4,
    limiter: RateLimiter | None = None,
) -> None:
    """Embebe lotes en paralelo (acotado por semáforo) y persiste cada uno al llegar."""
    limiter = limiter or RateLimiter(limits)
    semaphore = asyncio.Semaphore(concurrency)

    async def embed_batch(batch: range) -> tuple[range, np.ndarray]:
        async with semaphore:
            await limiter.acquire(sum(plan.token_counts[i] for i in batch))
            return batch, await embedder.embed([plan.texts[i] for i in batch])

    tasks = [
        asyncio.create_task(embed_batch(b))
        for b in batches(plan.token_counts, max_tokens=limits.batch_tokens)
    ]
    try:
        for done in asyncio.as_completed(tasks):
            batch, vectors = await done
            # Las escrituras se serializan aquí; lo ya persistido sobrevive a un fallo.
            async with store.writing():
                await store.add_embeddings([plan.hashes[i] for i in batch], vectors)
            if on_batch is not None:
                on_batch(len(batch))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
