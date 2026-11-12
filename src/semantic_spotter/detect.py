import heapq
import itertools
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from semantic_spotter.extractor import SHINGLE_SIZE
from semantic_spotter.models import Snippet

_CHUNK_ROWS = 2048
# Clases de equivalencia T2 más grandes que esto se conectan en estrella, no O(k²).
_MAX_FULL_CLASS = 50

# Tipos de clon (Roy & Cordy): 1 copia exacta, 2 renombrado (mismos tokens
# normalizados), 3 casi idéntico (Jaccard de k-gramas), 4 solo semántico (embedding).
CLONE_TYPES = {1: "Copias exactas", 2: "Renombrados", 3: "Casi idénticos", 4: "Semánticos"}


@dataclass(frozen=True, slots=True)
class Pair:
    a: int
    b: int
    score: float
    clone_type: int = 4


@dataclass(frozen=True)
class Group:
    members: list[int]
    pairs: list[Pair]
    # similitud máxima de cada miembro con otro miembro del grupo
    best_score: dict[int, float]
    mean_score: float
    generic: bool
    # el tipo del par menos parecido del grupo
    clone_type: int
    # líneas que se borrarían dejando solo la copia más grande
    deletable_lines: int
    # todos los miembros son bloques internos, no funciones
    blocks_only: bool
    # métodos homónimos en distintos archivos que no son copia literal:
    # probablemente implementaciones de una misma interfaz
    interface_like: bool

    @property
    def actionable(self) -> bool:
        return not (self.blocks_only or self.interface_like)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _ids(values: Sequence[str]) -> np.ndarray:
    index = {v: i for i, v in enumerate(dict.fromkeys(values))}
    return np.array([index[v] for v in values])


class _Positions:
    """Arrays por snippet para enmascarar pares en bloque con numpy."""

    def __init__(self, snippets: Sequence[Snippet]) -> None:
        self.files = _ids([s.file_path for s in snippets])
        self.hashes = _ids([s.content_hash for s in snippets])
        self.starts = np.array([s.start_byte for s in snippets])
        self.ends = np.array([s.end_byte for s in snippets])

    def overlapping(self, rows: np.ndarray) -> np.ndarray:
        """Máscara (len(rows), n): misma posición o una unidad dentro de la otra."""
        return (
            (self.files[rows, None] == self.files[None, :])
            & (self.starts[rows, None] < self.ends[None, :])
            & (self.starts[None, :] < self.ends[rows, None])
        )


def _similarity_chunks(snippets: Sequence[Snippet], matrix: np.ndarray, *, skip_identical: bool):
    """Produce (fila_inicial, similitudes) del triángulo superior por bloques.

    Las posiciones de la diagonal hacia abajo y los pares solapados (una unidad
    dentro de la otra en el mismo archivo) quedan en -inf; con `skip_identical`
    también los pares con el mismo contenido.
    """
    m = _normalize(matrix.astype(np.float32))
    n = len(snippets)
    pos = _Positions(snippets)
    cols = np.arange(n)
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        sims = m[start:stop] @ m.T
        rows = np.arange(start, stop)
        invalid = (cols[None, :] <= rows[:, None]) | pos.overlapping(rows)
        if skip_identical:
            invalid |= pos.hashes[rows, None] == pos.hashes[None, :]
        sims[invalid] = -np.inf
        yield start, sims


def candidate_pairs(
    snippets: Sequence[Snippet],
    matrix: np.ndarray,
    *,
    threshold: float | None = None,
    top: int | None = None,
    skip_identical: bool = False,
) -> list[Pair]:
    """Pares con similitud ≥ threshold, o los `top` más altos si se pide."""
    if len(snippets) < 2:
        return []
    pairs: list[Pair] = []
    heap: list[tuple[float, int, int]] = []
    chunks = _similarity_chunks(snippets, matrix, skip_identical=skip_identical)
    for start, sims in chunks:
        if top is not None:
            k = min(top, sims.size)
            flat = np.argpartition(sims, -k, axis=None)[-k:]
            for idx in flat:
                i, j = np.unravel_index(idx, sims.shape)
                score = float(sims[i, j])
                if score == -np.inf:
                    continue
                item = (score, start + int(i), int(j))
                if len(heap) < top:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)
        else:
            assert threshold is not None
            ii, jj = np.nonzero(sims >= threshold)
            pairs.extend(
                Pair(start + int(i), int(j), float(sims[i, j])) for i, j in zip(ii, jj)
            )
    if top is not None:
        pairs = [Pair(a, b, s) for s, a, b in heap]
    return sorted(pairs, key=lambda p: -p.score)


def renamed_pairs(
    snippets: Sequence[Snippet], matrix: np.ndarray, *, min_tokens: int = 40
) -> list[Pair]:
    """Pares con los mismos tokens normalizados (T2), con su coseno como score.

    Complementa al umbral de coseno, porque los embeddings son sensibles al
    renombrado. Se ignoran unidades de menos de `min_tokens` tokens, que suelen
    coincidir por pura forma (getters, delegaciones).
    """
    min_shingles = max(1, min_tokens - SHINGLE_SIZE + 1)
    classes: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(snippets):
        if s.norm_hash and len(s.shingles) >= min_shingles:
            classes[s.norm_hash].append(i)
    m = _normalize(matrix.astype(np.float32))
    pairs: list[Pair] = []
    for members in classes.values():
        if len(members) <= _MAX_FULL_CLASS:
            combos: Iterable[tuple[int, int]] = itertools.combinations(members, 2)
        else:
            combos = ((members[0], j) for j in members[1:])
        pairs.extend(
            Pair(i, j, float(m[i] @ m[j]))
            for i, j in combos
            if not snippets[i].overlaps(snippets[j])
        )
    return pairs


def _ordered(i: int, j: int, score: float) -> Pair:
    return Pair(min(i, j), max(i, j), score)


def query_pairs(
    snippets: Sequence[Snippet], matrix: np.ndarray, queries: Sequence[int], *, threshold: float
) -> list[Pair]:
    """Pares (consulta, cualquiera) con coseno ≥ threshold, sin solapados."""
    m = _normalize(matrix.astype(np.float32))
    pos = _Positions(snippets)
    cols = np.arange(len(snippets))
    found: dict[tuple[int, int], Pair] = {}
    for start in range(0, len(queries), _CHUNK_ROWS):
        rows = np.asarray(queries[start : start + _CHUNK_ROWS])
        sims = m[rows] @ m.T
        sims[(cols[None, :] == rows[:, None]) | pos.overlapping(rows)] = -np.inf
        for i, j in zip(*np.nonzero(sims >= threshold)):
            pair = _ordered(int(rows[i]), int(j), float(sims[i, j]))
            found.setdefault((pair.a, pair.b), pair)
    return sorted(found.values(), key=lambda p: -p.score)


class _ShingleIndex:
    """Índice invertido de shingles para calcular el Jaccard de una consulta contra todos."""

    def __init__(self, snippets: Sequence[Snippet], eligible: Sequence[int]) -> None:
        self.n = len(snippets)
        self.lengths = np.array([len(s.shingles) for s in snippets])
        flat = np.fromiter(
            itertools.chain.from_iterable(snippets[i].shingles for i in eligible), dtype=np.int64
        )
        owners = np.repeat(np.asarray(eligible, dtype=np.int64), self.lengths[list(eligible)])
        order = np.argsort(flat, kind="stable")
        self.flat, self.owners = flat[order], owners[order]

    def jaccard(self, shingles: Sequence[int]) -> np.ndarray:
        values = np.asarray(shingles, dtype=np.int64)
        lo = np.searchsorted(self.flat, values, "left")
        hi = np.searchsorted(self.flat, values, "right")
        shared = np.bincount(
            np.concatenate([self.owners[a:b] for a, b in zip(lo, hi)] or [np.empty(0, np.int64)]),
            minlength=self.n,
        )
        union = len(values) + self.lengths - shared
        return np.divide(shared, union, out=np.zeros(self.n), where=shared > 0)


def structural_pairs(
    snippets: Sequence[Snippet],
    queries: Sequence[int],
    matrix: np.ndarray | None = None,
    *,
    t3_jaccard: float,
    min_tokens: int = 40,
) -> list[Pair]:
    """Pares (consulta, cualquiera) T1, T2 o T3 encontrados sin embeddings.

    T1 por `content_hash`; T2 por `norm_hash` y T3 por Jaccard de shingles, ambos
    solo entre unidades de al menos `min_tokens` tokens (ver `renamed_pairs`).
    El score es el coseno si hay matriz y, si no, el Jaccard.
    """
    min_shingles = max(1, min_tokens - SHINGLE_SIZE + 1)
    eligible = [i for i, s in enumerate(snippets) if len(s.shingles) >= min_shingles]
    eligible_set = set(eligible)
    by_hash: dict[str, list[int]] = defaultdict(list)
    by_norm: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(snippets):
        by_hash[s.content_hash].append(i)
    for i in eligible:
        by_norm[snippets[i].norm_hash].append(i)
    index = _ShingleIndex(snippets, eligible) if eligible else None
    m = _normalize(matrix.astype(np.float32)) if matrix is not None else None

    found: dict[tuple[int, int], Pair] = {}
    for q in queries:
        sq = snippets[q]
        jaccard: np.ndarray | None = None
        candidates = set(by_hash[sq.content_hash])
        if q in eligible_set and index is not None:
            candidates.update(by_norm[sq.norm_hash])
            jaccard = index.jaccard(sq.shingles)
            candidates.update(int(j) for j in np.nonzero(jaccard >= t3_jaccard)[0])
        for j in candidates:
            if j == q or sq.overlaps(snippets[j]):
                continue
            if m is not None:
                score = float(m[q] @ m[j])
            elif jaccard is not None:
                score = float(jaccard[j])
            else:
                score = 1.0
            pair = _ordered(q, j, score)
            found.setdefault((pair.a, pair.b), pair)
    return sorted(found.values(), key=lambda p: -p.score)


def changed_occurrences(base: Sequence[Snippet], head: Sequence[Snippet]) -> list[int]:
    """Índices de `head` que introduce el cambio respecto de `base`.

    Un `content_hash` cambió si aparece más veces en head que en base: contenido
    nuevo, editado o copiado. Mover o renombrar un archivo y correr líneas no
    cambia ese conteo. De un hash cambiado se toman las ocurrencias de las rutas
    donde su conteo subió, es decir, la copia y no el original.
    """
    base_hash = Counter(s.content_hash for s in base)
    head_hash = Counter(s.content_hash for s in head)
    base_path = Counter((s.file_path, s.content_hash) for s in base)
    head_path = Counter((s.file_path, s.content_hash) for s in head)
    return [
        i
        for i, s in enumerate(head)
        if head_hash[s.content_hash] > base_hash[s.content_hash]
        and head_path[(s.file_path, s.content_hash)] > base_path[(s.file_path, s.content_hash)]
    ]


def merge_pairs(*pair_lists: Iterable[Pair]) -> list[Pair]:
    """Une listas de pares sin repetir (a, b), ordenadas por score."""
    merged: dict[tuple[int, int], Pair] = {}
    for pair in itertools.chain(*pair_lists):
        merged.setdefault((pair.a, pair.b), pair)
    return sorted(merged.values(), key=lambda p: -p.score)


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def clone_type(a: Snippet, b: Snippet, *, t3_jaccard: float) -> int:
    if a.content_hash == b.content_hash:
        return 1
    if a.norm_hash and a.norm_hash == b.norm_hash:
        return 2
    if _jaccard(a.shingles, b.shingles) >= t3_jaccard:
        return 3
    return 4


def classify_pairs(
    snippets: Sequence[Snippet], pairs: Sequence[Pair], *, t3_jaccard: float = 0.7
) -> list[Pair]:
    return [
        replace(p, clone_type=clone_type(snippets[p.a], snippets[p.b], t3_jaccard=t3_jaccard))
        for p in pairs
    ]


def suppress_subsumed(snippets: Sequence[Snippet], pairs: Sequence[Pair]) -> list[Pair]:
    """Descarta (a, b) si ya se aceptó (A, B) con a ⊆ A y b ⊆ B (o cruzado).

    Se procesan primero los pares más grandes para que gane el clon de mayor
    alcance (p. ej. dos funciones completas frente a sus `for` internos).
    """
    accepted: list[Pair] = []
    by_files: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    ordered = sorted(
        pairs, key=lambda p: (-(snippets[p.a].size + snippets[p.b].size), -p.score)
    )
    for pair in ordered:
        a, b = snippets[pair.a], snippets[pair.b]
        dominated = any(
            (snippets[o.a].contains(a) and snippets[o.b].contains(b))
            or (snippets[o.a].contains(b) and snippets[o.b].contains(a))
            for o in by_files[(a.file_path, b.file_path)] + by_files[(b.file_path, a.file_path)]
        )
        if dominated:
            continue
        accepted.append(pair)
        by_files[(a.file_path, b.file_path)].append(pair)
    return sorted(accepted, key=lambda p: -p.score)


def group_pairs(
    snippets: Sequence[Snippet], pairs: Sequence[Pair], *, max_group: int
) -> list[Group]:
    """Componentes conexas (union-find) sobre los pares aceptados."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        parent[find(p.a)] = find(p.b)

    components: dict[int, list[Pair]] = defaultdict(list)
    for p in pairs:
        components[find(p.a)].append(p)

    groups = []
    for component in components.values():
        best: dict[int, float] = {}
        for p in component:
            best[p.a] = max(best.get(p.a, -1.0), p.score)
            best[p.b] = max(best.get(p.b, -1.0), p.score)
        members = sorted(best)
        lines = [snippets[i].lines for i in members]
        kind = max(p.clone_type for p in component)
        groups.append(
            Group(
                members=members,
                pairs=sorted(component, key=lambda p: -p.score),
                best_score=best,
                mean_score=float(np.mean([p.score for p in component])),
                generic=len(members) > max_group,
                clone_type=kind,
                deletable_lines=sum(lines) - max(lines),
                blocks_only=all(snippets[i].unit == "block" for i in members),
                interface_like=kind > 2 and _same_method([snippets[i] for i in members]),
            )
        )
    # Primero lo accionable, luego probables interfaces y al final solo bloques;
    # dentro de cada categoría, por líneas eliminables ponderadas por similitud.
    return sorted(
        groups,
        key=lambda g: (g.blocks_only, g.interface_like, -(g.deletable_lines * g.mean_score)),
    )


def drop_ignored(
    snippets: Sequence[Snippet], groups: Sequence[Group], changed: Collection[int] | None = None
) -> list[Group]:
    """Quita los grupos cuyos miembros tienen todos marca de ignorar.

    Con `changed` (en `check`) solo cuentan los miembros cambiados: el agente no
    tiene que marcar código que no tocó, y una copia nueva sin marca de una
    función marcada sigue reportándose.
    """

    def ignored(g: Group) -> bool:
        judged = [i for i in g.members if changed is None or i in changed]
        return bool(judged) and all(snippets[i].ignore_reason for i in judged)

    return [g for g in groups if not ignored(g)]


def _same_method(members: Sequence[Snippet]) -> bool:
    """Todos son métodos con el mismo nombre, cada uno en un archivo distinto."""
    if not all(s.kind == "method" for s in members):
        return False
    names = {s.name.rsplit(".", 1)[-1] for s in members}
    return len(names) == 1 and len({s.file_path for s in members}) == len(members)
