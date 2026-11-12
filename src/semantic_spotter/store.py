import asyncio
import fcntl
import itertools
import os
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Self

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.db import AsyncConnection
from lancedb.table import AsyncTable

from semantic_spotter import gitsource
from semantic_spotter.models import Snippet, TreeFile

# Nombre del índice dentro del directorio común de git.
DB_DIRNAME = "semantic-spotter"
# Si está definida, reemplaza la ruta por defecto del índice.
DB_ENV = "SEMANTIC_SPOTTER_DB"
# Archivo dentro del índice sobre el que se toma el `flock` de escritura.
_LOCK_FILENAME = "write.lock"
_LOCK_POLL_SECONDS = 0.02

# Ningún esquema guarda el texto de los snippets: se recupera de los blobs de git.
# El sufijo cambia cuando cambia el esquema; las tablas viejas quedan sin uso.
_SUFFIX = "_v4"

# commit → árbol: registra que el commit se analizó.
_COMMITS_SCHEMA = pa.schema(
    [pa.field("commit_sha", pa.string()), pa.field("tree_oid", pa.string())]
)
# Marca que un árbol quedó registrado completo en `tree_files`.
_TREES_SCHEMA = pa.schema(
    [pa.field("tree_oid", pa.string()), pa.field("file_count", pa.int64())]
)
# Archivos de lenguajes soportados de cada árbol.
_TREE_FILES_SCHEMA = pa.schema(
    [
        pa.field("tree_oid", pa.string()),
        pa.field("path", pa.string()),
        pa.field("blob_oid", pa.string()),
        pa.field("language", pa.string()),
    ]
)
# Marca que un (blob, lenguaje, config) quedó extraído por completo.
_BLOBS_SCHEMA = pa.schema(
    [
        pa.field("blob_oid", pa.string()),
        pa.field("language", pa.string()),
        pa.field("extract_key", pa.string()),
        pa.field("generated", pa.bool_()),
        pa.field("snippet_count", pa.int64()),
    ]
)
# Ubicación de cada snippet dentro de su blob, con el motivo de su marca de
# ignorar efectiva ("" sin marca); es el único texto del código que se guarda.
_BLOB_SNIPPETS_SCHEMA = pa.schema(
    [
        pa.field("blob_oid", pa.string()),
        pa.field("language", pa.string()),
        pa.field("extract_key", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("unit", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("name", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("start_byte", pa.int64()),
        pa.field("end_byte", pa.int64()),
        pa.field("ignore_reason", pa.string()),
    ]
)
# Huella estructural por contenido; `normalizer` versiona la normalización.
_CONTENTS_SCHEMA = pa.schema(
    [
        pa.field("content_hash", pa.string()),
        pa.field("normalizer", pa.string()),
        pa.field("norm_hash", pa.string()),
        pa.field("shingles", pa.list_(pa.int64())),
    ]
)

# Commits recorridos por `index`, por config de extracción.
_HISTORY_SCHEMA = pa.schema(
    [
        pa.field("commit_sha", pa.string()),
        pa.field("extract_key", pa.string()),
        pa.field("committed_at", pa.int64()),
    ]
)
# Primer commit recorrido en el que cada (ruta, contenido) apareció.
_FIRST_SEEN_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("extract_key", pa.string()),
        pa.field("commit_sha", pa.string()),
        pa.field("committed_at", pa.int64()),
    ]
)

_LOCATION_FIELDS = [
    f.name for f in _BLOB_SNIPPETS_SCHEMA if f.name not in ("language", "extract_key")
]
# Máximo de valores por cláusula `IN (...)`.
_IN_CHUNK = 1000


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _in(column: str, values: Sequence[str]) -> str:
    return f"{column} IN ({', '.join(_sql_str(v) for v in values)})"


def _chunks(values: Iterable[str]) -> Iterable[list[str]]:
    return (list(c) for c in itertools.batched(values, _IN_CHUNK))


@asynccontextmanager
async def _flock(lock_path: Path) -> AsyncIterator[None]:
    """`flock` exclusivo sobre `lock_path` (macOS y Linux), sin bloquear el event loop."""
    with open(lock_path, "a") as lock:
        # Se reintenta sin bloquear hasta que el otro proceso suelte el lock.
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class Sighting:
    """Dónde y cuándo se vio por primera vez un (ruta, contenido)."""

    path: str
    content_hash: str
    commit_sha: str
    committed_at: int


@dataclass(frozen=True)
class ExtractedBlob:
    blob_oid: str
    language: str
    generated: bool
    snippets: Sequence[Snippet]


async def index_dir(repo_root: Path) -> Path:
    """`$SEMANTIC_SPOTTER_DB` o `<directorio común de git>/semantic-spotter/`.

    El directorio común es el mismo para todos los worktrees, así que un
    worktree nuevo encuentra ya extraído lo que extrajeron los demás.
    """
    if override := os.environ.get(DB_ENV):
        return Path(override)
    return await gitsource.common_dir(repo_root) / DB_DIRNAME


class Store:
    """Índice compartido por los worktrees del repo (ver `index_dir`).

    - `commits`: commit → árbol, para los commits analizados.
    - `trees` + `tree_files`: (ruta, blob) de cada árbol registrado.
    - `blobs` + `blob_snippets`: snippets de cada blob por config de extracción;
      un blob se extrae una sola vez aunque aparezca en muchos commits.
    - `contents`: `norm_hash` y shingles por `content_hash`.
    - `history` + `first_seen`: commits recorridos por `index` y el primero en
      que apareció cada (ruta, contenido).
    - `embeddings__<modelo>__<dim>`: vector por `content_hash`.
    """

    def __init__(
        self, db: AsyncConnection, tables: dict[str, AsyncTable], dim: int, lock_path: Path
    ) -> None:
        self._db = db
        self._lock_path = lock_path
        self._commits = tables["commits"]
        self._trees = tables["trees"]
        self._tree_files = tables["tree_files"]
        self._blobs = tables["blobs"]
        self._blob_snippets = tables["blob_snippets"]
        self._contents = tables["contents"]
        self._history = tables["history"]
        self._first_seen = tables["first_seen"]
        self._embeddings = tables["embeddings"]
        self.dim = dim

    @classmethod
    async def open(cls, repo_root: Path, model: str, dim: int) -> Self:
        path = await index_dir(repo_root)
        path.mkdir(parents=True, exist_ok=True)
        # Consistencia fuerte: cada lectura ve lo último escrito por cualquier handle.
        db = await lancedb.connect_async(path, read_consistency_interval=timedelta(0))
        embeddings_schema = pa.schema(
            [
                pa.field("content_hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        schemas = {
            "commits" + _SUFFIX: _COMMITS_SCHEMA,
            "trees" + _SUFFIX: _TREES_SCHEMA,
            "tree_files" + _SUFFIX: _TREE_FILES_SCHEMA,
            "blobs" + _SUFFIX: _BLOBS_SCHEMA,
            "blob_snippets" + _SUFFIX: _BLOB_SNIPPETS_SCHEMA,
            "contents" + _SUFFIX: _CONTENTS_SCHEMA,
            "history" + _SUFFIX: _HISTORY_SCHEMA,
            "first_seen" + _SUFFIX: _FIRST_SEEN_SCHEMA,
            "embeddings__" + re.sub(r"\W", "_", model) + f"__{dim}": embeddings_schema,
        }
        lock_path = path / _LOCK_FILENAME
        tables = {}
        # Crear las tablas también es escribir: dos primeras corridas a la vez no chocan.
        async with _flock(lock_path):
            for name, schema in schemas.items():
                key = name.removesuffix(_SUFFIX) if name.endswith(_SUFFIX) else "embeddings"
                tables[key] = await db.create_table(name, schema=schema, exist_ok=True)
        return cls(db, tables, dim, lock_path)

    @asynccontextmanager
    async def writing(self) -> AsyncIterator[None]:
        """Lock exclusivo de escritura sobre el índice, compartido entre procesos.

        Toda escritura (`save_*`, `add_embeddings`) corre dentro de un `writing()`
        explícito; esos métodos no toman el lock, y `writing()` no es reentrante.
        Cada uno vuelve a consultar, ya con el lock, qué falta escribir. Las
        lecturas no lo necesitan.
        """
        async with _flock(self._lock_path):
            yield

    async def _rows(
        self, table: AsyncTable, where: str, columns: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        query = table.query().where(where)
        if columns is not None:
            query = query.select(list(columns))
        return (await query.to_arrow()).to_pylist()

    # --- commits y árboles ----------------------------------------------------

    async def link_commit(self, sha: str, tree_oid: str) -> None:
        if await self._commits.count_rows(f"commit_sha = {_sql_str(sha)}"):
            return
        await self._commits.add(
            pa.Table.from_pylist(
                [{"commit_sha": sha, "tree_oid": tree_oid}], schema=_COMMITS_SCHEMA
            )
        )

    async def has_tree(self, tree_oid: str) -> bool:
        return await self._trees.count_rows(f"tree_oid = {_sql_str(tree_oid)}") > 0

    async def save_tree(self, tree_oid: str, files: Sequence[TreeFile]) -> None:
        if await self.has_tree(tree_oid):
            return  # otro proceso lo registró mientras esperábamos el lock
        where = f"tree_oid = {_sql_str(tree_oid)}"
        # Limpia restos de un registro interrumpido antes de escribir.
        await self._tree_files.delete(where)
        if files:
            rows = [
                {"tree_oid": tree_oid, "path": f.path, "blob_oid": f.blob_oid, "language": f.language}
                for f in files
            ]
            await self._tree_files.add(pa.Table.from_pylist(rows, schema=_TREE_FILES_SCHEMA))
        await self._trees.add(
            pa.Table.from_pylist(
                [{"tree_oid": tree_oid, "file_count": len(files)}], schema=_TREES_SCHEMA
            )
        )

    async def load_tree(self, tree_oid: str) -> list[TreeFile]:
        rows = await self._rows(
            self._tree_files, f"tree_oid = {_sql_str(tree_oid)}", ["path", "blob_oid", "language"]
        )
        return sorted((TreeFile(**r) for r in rows), key=lambda f: f.path)

    # --- blobs ----------------------------------------------------------------

    async def missing_blobs(
        self, blobs: Iterable[tuple[str, str]], extract_key: str
    ) -> set[tuple[str, str]]:
        """(blob_oid, lenguaje) que aún no se extrajeron con esta config."""
        wanted = set(blobs)
        done: set[tuple[str, str]] = set()
        key = f"extract_key = {_sql_str(extract_key)}"
        for chunk in _chunks(sorted({oid for oid, _ in wanted})):
            rows = await self._rows(
                self._blobs, f"{key} AND {_in('blob_oid', chunk)}", ["blob_oid", "language"]
            )
            done.update((r["blob_oid"], r["language"]) for r in rows)
        return wanted - done

    async def save_blobs(
        self, extract_key: str, normalizer: str, blobs: Sequence[ExtractedBlob]
    ) -> None:
        """Guarda ubicaciones y huellas; la marca en `blobs` se escribe al final."""
        if not blobs:
            return
        key = f"extract_key = {_sql_str(extract_key)}"
        for chunk in _chunks(sorted({b.blob_oid for b in blobs})):
            await self._blob_snippets.delete(f"{key} AND {_in('blob_oid', chunk)}")
        locations = [
            {
                "blob_oid": b.blob_oid,
                "language": b.language,
                "extract_key": extract_key,
                **{f: getattr(s, f) for f in _LOCATION_FIELDS if f != "blob_oid"},
            }
            for b in blobs
            for s in b.snippets
        ]
        if locations:
            await self._blob_snippets.add(
                pa.Table.from_pylist(locations, schema=_BLOB_SNIPPETS_SCHEMA)
            )
        await self._add_contents(normalizer, [s for b in blobs for s in b.snippets])
        await self._blobs.add(
            pa.Table.from_pylist(
                [
                    {
                        "blob_oid": b.blob_oid,
                        "language": b.language,
                        "extract_key": extract_key,
                        "generated": b.generated,
                        "snippet_count": len(b.snippets),
                    }
                    for b in blobs
                ],
                schema=_BLOBS_SCHEMA,
            )
        )

    async def _add_contents(self, normalizer: str, snippets: Sequence[Snippet]) -> None:
        by_hash = {s.content_hash: s for s in snippets}
        existing = await self._load_contents(by_hash, normalizer)
        rows = [
            {
                "content_hash": h,
                "normalizer": normalizer,
                "norm_hash": s.norm_hash,
                "shingles": list(s.shingles),
            }
            for h, s in by_hash.items()
            if h not in existing
        ]
        if rows:
            await self._contents.add(pa.Table.from_pylist(rows, schema=_CONTENTS_SCHEMA))

    async def _load_contents(
        self, hashes: Iterable[str], normalizer: str
    ) -> dict[str, tuple[str, tuple[int, ...]]]:
        out: dict[str, tuple[str, tuple[int, ...]]] = {}
        norm = f"normalizer = {_sql_str(normalizer)}"
        for chunk in _chunks(sorted(set(hashes))):
            rows = await self._rows(
                self._contents,
                f"{norm} AND {_in('content_hash', chunk)}",
                ["content_hash", "norm_hash", "shingles"],
            )
            out.update((r["content_hash"], (r["norm_hash"], tuple(r["shingles"]))) for r in rows)
        return out

    async def load_snippets(
        self, files: Sequence[TreeFile], extract_key: str, normalizer: str
    ) -> list[Snippet]:
        """Snippets de los archivos dados (ya extraídos), unidos con su huella."""
        key = f"extract_key = {_sql_str(extract_key)}"
        by_blob: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for chunk in _chunks(sorted({f.blob_oid for f in files})):
            rows = await self._rows(
                self._blob_snippets,
                f"{key} AND {_in('blob_oid', chunk)}",
                [*_LOCATION_FIELDS, "language"],
            )
            for r in rows:
                by_blob[(r["blob_oid"], r["language"])].append(r)
        contents = await self._load_contents(
            (r["content_hash"] for rows in by_blob.values() for r in rows), normalizer
        )
        snippets = []
        for f in files:
            for r in by_blob.get((f.blob_oid, f.language), []):
                norm_hash, shingles = contents.get(r["content_hash"], ("", ()))
                snippets.append(
                    Snippet(
                        **r | {"file_path": f.path, "norm_hash": norm_hash, "shingles": shingles}
                    )
                )
        return sorted(snippets, key=lambda s: (s.file_path, s.start_byte, s.end_byte))

    # --- historial ------------------------------------------------------------

    async def walked_commits(self, extract_key: str) -> set[str]:
        rows = await self._rows(
            self._history, f"extract_key = {_sql_str(extract_key)}", ["commit_sha"]
        )
        return {r["commit_sha"] for r in rows}

    async def seen_keys(self, extract_key: str) -> set[tuple[str, str]]:
        rows = await self._rows(
            self._first_seen, f"extract_key = {_sql_str(extract_key)}", ["path", "content_hash"]
        )
        return {(r["path"], r["content_hash"]) for r in rows}

    async def save_history(
        self,
        extract_key: str,
        commits: Sequence[tuple[str, int]],
        sightings: Sequence[Sighting],
    ) -> None:
        """Las marcas de commits van al final: un corte deja commits por recorrer, no huecos."""
        if sightings:
            rows = [
                {
                    "path": x.path,
                    "content_hash": x.content_hash,
                    "extract_key": extract_key,
                    "commit_sha": x.commit_sha,
                    "committed_at": x.committed_at,
                }
                for x in sightings
            ]
            await self._first_seen.add(pa.Table.from_pylist(rows, schema=_FIRST_SEEN_SCHEMA))
        if commits:
            rows = [
                {"commit_sha": sha, "extract_key": extract_key, "committed_at": ts}
                for sha, ts in commits
            ]
            await self._history.add(pa.Table.from_pylist(rows, schema=_HISTORY_SCHEMA))

    async def first_seen(
        self, snippets: Sequence[Snippet], extract_key: str
    ) -> dict[tuple[str, str], Sighting]:
        """Primer avistamiento conocido de cada (ruta, contenido) de los snippets dados."""
        wanted = {(s.file_path, s.content_hash) for s in snippets}
        out: dict[tuple[str, str], Sighting] = {}
        key = f"extract_key = {_sql_str(extract_key)}"
        for chunk in _chunks(sorted({h for _, h in wanted})):
            rows = await self._rows(
                self._first_seen,
                f"{key} AND {_in('content_hash', chunk)}",
                ["path", "content_hash", "commit_sha", "committed_at"],
            )
            for r in rows:
                k = (r["path"], r["content_hash"])
                if k in wanted and (k not in out or r["committed_at"] < out[k].committed_at):
                    out[k] = Sighting(**r)
        return out

    # --- embeddings -----------------------------------------------------------

    async def _cached_hashes(self) -> set[str]:
        table = await self._embeddings.query().select(["content_hash"]).to_arrow()
        return set(table.column("content_hash").to_pylist())

    async def missing_hashes(self, hashes: Sequence[str]) -> list[str]:
        cached = await self._cached_hashes()
        return [h for h in dict.fromkeys(hashes) if h not in cached]

    async def add_embeddings(self, hashes: Sequence[str], vectors: np.ndarray) -> None:
        # Otro proceso pudo haber guardado algunos mientras se embebían.
        cached = await self._cached_hashes()
        keep = [i for i, h in enumerate(hashes) if h not in cached]
        if not keep:
            return
        flat = pa.array(vectors[keep].astype(np.float32).ravel())
        await self._embeddings.add(
            pa.table(
                {
                    "content_hash": pa.array([hashes[i] for i in keep], pa.string()),
                    "vector": pa.FixedSizeListArray.from_arrays(flat, self.dim),
                }
            )
        )

    async def load_vectors(self, hashes: Sequence[str]) -> np.ndarray:
        """Matriz alineada con `hashes`; falla si alguno no está en caché."""
        table = await self._embeddings.to_arrow()
        index = {h: i for i, h in enumerate(table.column("content_hash").to_pylist())}
        all_vectors = (
            table.column("vector").combine_chunks().flatten().to_numpy().reshape(-1, self.dim)
        )
        missing = [h for h in hashes if h not in index]
        if missing:
            raise KeyError(f"{len(missing)} embeddings faltantes en caché")
        return all_vectors[[index[h] for h in hashes]]
