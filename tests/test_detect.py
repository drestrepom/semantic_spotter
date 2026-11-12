from dataclasses import replace

import numpy as np

from semantic_spotter.detect import (
    Pair,
    candidate_pairs,
    changed_occurrences,
    classify_pairs,
    group_pairs,
    merge_pairs,
    query_pairs,
    renamed_pairs,
    structural_pairs,
    suppress_subsumed,
)
from semantic_spotter.models import Snippet


def snip(path, start, end, name="", unit="function", kind="func", **extra):
    return Snippet(
        content_hash=f"{path}:{start}", blob_oid="b", file_path=path, language="go",
        unit=unit, kind=kind, name=name, start_line=start, end_line=end,
        start_byte=start, end_byte=end, **extra,
    )


def test_overlapping_pairs_are_never_candidates():
    snippets = [snip("a.go", 0, 100), snip("a.go", 10, 50, unit="block")]
    matrix = np.ones((2, 4), dtype=np.float32)
    assert candidate_pairs(snippets, matrix, threshold=0.5) == []
    assert candidate_pairs(snippets, matrix, top=5) == []


def test_threshold_and_top():
    snippets = [snip("a.go", 0, 10), snip("b.go", 0, 10), snip("c.go", 0, 10)]
    matrix = np.array([[1, 0], [1, 0.1], [0, 1]], dtype=np.float32)
    pairs = candidate_pairs(snippets, matrix, threshold=0.9)
    assert [(p.a, p.b) for p in pairs] == [(0, 1)]
    top = candidate_pairs(snippets, matrix, top=2)
    assert [(p.a, p.b) for p in top] == [(0, 1), (1, 2)]


def test_subsumption_keeps_largest_pair():
    f1, f1_for = snip("a.go", 0, 100), snip("a.go", 10, 60, unit="block")
    f2, f2_for = snip("b.go", 0, 100), snip("b.go", 20, 70, unit="block")
    helper = snip("c.go", 0, 50)
    snippets = [f1, f1_for, f2, f2_for, helper]
    matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32)
    pairs = suppress_subsumed(snippets, candidate_pairs(snippets, matrix, threshold=0.99))
    got = {(snippets[p.a].location, snippets[p.b].location) for p in pairs}
    # (f1_for, f2_for) queda subsumido por (f1, f2); los pares con el helper sobreviven.
    assert got == {
        ("a.go:0-100", "b.go:0-100"),
        ("a.go:10-60", "c.go:0-50"),
        ("b.go:20-70", "c.go:0-50"),
    }


def test_group_pairs_union_find_and_generic():
    snippets = [snip(f"{i}.go", 0, 10) for i in range(4)]
    matrix = np.array([[1, 0], [1, 0.01], [1, 0.02], [0, 1]], dtype=np.float32)
    pairs = candidate_pairs(snippets, matrix, threshold=0.99)
    groups = group_pairs(snippets, pairs, max_group=2)
    assert len(groups) == 1
    assert groups[0].members == [0, 1, 2]
    assert groups[0].generic
    assert groups[0].clone_type == 4


def test_skip_identical_and_exact_groups():
    snippets = [
        replace(snip("a.go", 0, 10), content_hash="same"),
        replace(snip("b.go", 0, 10), content_hash="same"),
        replace(snip("c.go", 0, 10), content_hash="other"),
    ]
    matrix = np.array([[1, 0], [1, 0], [1, 0.05]], dtype=np.float32)
    top = candidate_pairs(snippets, matrix, top=5, skip_identical=True)
    assert {(p.a, p.b) for p in top} == {(0, 2), (1, 2)}

    exact_pair = [p for p in candidate_pairs(snippets, matrix, threshold=0.999) if p.b == 1]
    groups = group_pairs(snippets, classify_pairs(snippets, exact_pair), max_group=8)
    assert groups[0].clone_type == 1


def test_classify_pairs_t1_to_t4():
    base = snip("a.go", 0, 10, norm_hash="n", shingles=(1, 2, 3, 4))
    snippets = [
        base,
        replace(base, file_path="b.go"),  # mismo contenido
        replace(base, file_path="c.go", content_hash="c"),  # mismos tokens normalizados
        replace(base, file_path="d.go", content_hash="d", norm_hash="x", shingles=(1, 2, 3, 5)),
        replace(base, file_path="e.go", content_hash="e", norm_hash="y", shingles=(9,)),
    ]
    pairs = [Pair(0, i, 0.9) for i in range(1, 5)]
    types = [p.clone_type for p in classify_pairs(snippets, pairs, t3_jaccard=0.5)]
    assert types == [1, 2, 3, 4]


def test_renamed_pairs_found_below_threshold():
    shingles = tuple(range(50))
    snippets = [
        snip("a.go", 0, 10, norm_hash="n", shingles=shingles),
        snip("b.go", 0, 10, norm_hash="n", shingles=shingles),
        snip("c.go", 0, 10, norm_hash="n", shingles=(1, 2)),  # demasiado corto
        snip("a.go", 2, 8, unit="block", norm_hash="n", shingles=shingles),  # solapado
    ]
    matrix = np.array([[1, 0], [0.6, 0.8], [1, 0], [1, 0]], dtype=np.float32)
    assert candidate_pairs(snippets, matrix, threshold=0.9) == [Pair(0, 2, 1.0), Pair(2, 3, 1.0)]
    renamed = renamed_pairs(snippets, matrix)
    assert [(p.a, p.b) for p in renamed] == [(0, 1), (1, 3)]
    assert renamed[0].score == np.float32(0.6)
    merged = merge_pairs(candidate_pairs(snippets, matrix, threshold=0.9), renamed)
    assert [(p.a, p.b) for p in merged] == [(0, 2), (2, 3), (0, 1), (1, 3)]


def test_group_metrics_and_ordering():
    funcs = [snip("a.go", 1, 30), snip("b.go", 1, 20)]
    blocks = [snip("c.go", 1, 50, unit="block"), snip("d.go", 1, 50, unit="block")]
    methods = [
        snip("x/impl.go", 1, 40, name="impl.Price", kind="method"),
        snip("y/impl.go", 1, 40, name="impl.Price", kind="method"),
    ]
    snippets = funcs + blocks + methods
    pairs = [Pair(4, 5, 0.95), Pair(2, 3, 0.95), Pair(0, 1, 0.92)]
    groups = group_pairs(snippets, pairs, max_group=8)
    assert [g.members for g in groups] == [[0, 1], [4, 5], [2, 3]]
    assert groups[0].deletable_lines == 20 and groups[0].actionable
    assert groups[1].interface_like and not groups[1].blocks_only
    assert groups[2].blocks_only and not groups[2].interface_like

    # Una copia literal de métodos homónimos sí es accionable.
    exact = replace(methods[1], content_hash=methods[0].content_hash)
    snippets[5] = exact
    group = group_pairs(snippets, classify_pairs(snippets, [Pair(4, 5, 1.0)]), max_group=8)[0]
    assert group.clone_type == 1 and group.actionable


def occ(path, start, content_hash):
    return replace(snip(path, start, start + 10), content_hash=content_hash)


def test_changed_occurrences():
    base = [occ("a.go", 0, "keep"), occ("a.go", 20, "edit"), occ("m.go", 0, "move")]
    head = [
        occ("a.go", 5, "keep"),  # corrida de línea
        occ("a.go", 25, "edited"),  # editada
        occ("n.go", 0, "move"),  # archivo renombrado
        occ("b.go", 0, "keep"),  # copiada a otro archivo
        occ("c.go", 0, "new"),  # nueva
    ]
    assert changed_occurrences(base, head) == [1, 3, 4]
    # Sin base, todo es nuevo.
    assert changed_occurrences([], head) == [0, 1, 2, 3, 4]


def test_structural_pairs_without_embeddings():
    long = tuple(range(100))
    snippets = [
        snip("q.go", 0, 10, norm_hash="n", shingles=long),  # consulta
        replace(snip("a.go", 0, 10, norm_hash="n", shingles=long), content_hash="q.go:0"),  # T1
        snip("b.go", 0, 10, norm_hash="n", shingles=long),  # T2
        snip("c.go", 0, 10, norm_hash="x", shingles=long[:90] + (500,)),  # T3
        snip("d.go", 0, 10, norm_hash="y", shingles=tuple(range(1000, 1100))),  # nada
        snip("q.go", 2, 8, unit="block", norm_hash="n", shingles=long),  # solapado
    ]
    pairs = structural_pairs(snippets, [0], t3_jaccard=0.8)
    assert {(p.a, p.b) for p in pairs} == {(0, 1), (0, 2), (0, 3)}
    types = {p.b: p.clone_type for p in classify_pairs(snippets, pairs, t3_jaccard=0.8)}
    assert types == {1: 1, 2: 2, 3: 3}
    # Sin matriz el score es el Jaccard; con matriz, el coseno.
    assert next(p for p in pairs if p.b == 3).score == 90 / 101
    matrix = np.eye(6, dtype=np.float32)
    assert all(p.score == 0 for p in structural_pairs(snippets, [0], matrix, t3_jaccard=0.8))

    # Las unidades cortas solo cuentan como copia exacta.
    short = [replace(s, shingles=(1, 2)) for s in snippets]
    assert {(p.a, p.b) for p in structural_pairs(short, [0], t3_jaccard=0.8)} == {(0, 1)}


def test_query_pairs_only_touch_queries():
    snippets = [snip("a.go", 0, 10), snip("b.go", 0, 10), snip("c.go", 0, 10)]
    matrix = np.array([[1, 0], [1, 0.05], [1, 0.1]], dtype=np.float32)
    pairs = query_pairs(snippets, matrix, [2], threshold=0.9)
    assert {(p.a, p.b) for p in pairs} == {(0, 2), (1, 2)}
    # Un par entre dos consultas sale una sola vez.
    both = query_pairs(snippets, matrix, [0, 2], threshold=0.9)
    assert sorted((p.a, p.b) for p in both) == [(0, 1), (0, 2), (1, 2)]
