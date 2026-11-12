from semantic_spotter.extractor import content_hash, extract, snippet_text
from semantic_spotter.languages import GO, TSX, TYPESCRIPT, spec_for

GO_SRC = b"""package x

func Sum(xs []int) int {
	total := 0
	for _, x := range xs {
		if x < 0 {
			continue
		}
		total += x
		total += 0
		total += 0
	}
	return total
}

func (s *Store[T]) Get(k string) int {
	if k == "" {
		return 0
	} else if k == "a" {
		return 1
	} else {
		return 2
	}
	f := func() int {
		return 3
	}
	return f()
}
"""


def _by_name(snippets):
    return {s.name: s for s in snippets}


def test_go_functions_methods_and_blocks():
    snippets = extract(GO, "x.go", GO_SRC, "sha", min_func_lines=3, min_block_lines=4)
    names = _by_name(snippets)
    assert set(names) == {"Sum", "Sum › for@L5", "Store.Get", "Store.Get › if@L17"}
    assert names["Sum"].kind == "func" and names["Store.Get"].kind == "method"
    assert (names["Sum"].start_line, names["Sum"].end_line) == (3, 14)
    # Los `else if` forman parte de la cadena, no son unidades aparte; el `func` anónimo no cuenta.
    assert sum(1 for s in snippets if s.kind == "if") == 1
    assert all(s.blob_oid == "sha" for s in snippets)


def test_block_text_is_dedented_and_hash_stable():
    snippets = _by_name(extract(GO, "x.go", GO_SRC, "a", min_block_lines=4))
    block = snippets["Sum › for@L5"]
    text = snippet_text(GO_SRC, block.start_byte, block.end_byte)
    assert text.startswith("for _, x := range xs {\n\tif x < 0 {")
    assert content_hash(text) == block.content_hash
    assert GO_SRC[block.start_byte : block.end_byte].startswith(b"for _, x")
    again = _by_name(extract(GO, "y.go", b"package y\n\n" + GO_SRC[11:], "b", min_block_lines=4))
    assert again["Sum › for@L5"].content_hash == block.content_hash


def test_min_lines_filters():
    snippets = extract(GO, "x.go", GO_SRC, "sha", min_func_lines=100, min_block_lines=100)
    assert snippets == []


TS_SRC = b"""
export function add(a: number, b: number) {
  const r = a + b;
  return r;
}

class Repo {
  find(id: string) {
    const x = id.trim();
    return x;
  }
  handler = (e: Event) => {
    console.log(e);
    return 1;
  };
}

const double = (x: number) => {
  const y = x * 2;
  return y;
};

[1, 2].map((v) => {
  const w = v + 1;
  return w;
});
"""


def test_typescript_named_functions_only():
    names = _by_name(extract(TYPESCRIPT, "a.ts", TS_SRC, "sha"))
    assert set(names) == {"add", "Repo.find", "Repo.handler", "double"}


def test_tsx_parses_jsx():
    src = b"const View = () => {\n  const x = 1;\n  return <div>{x}</div>;\n};\n"
    assert [s.name for s in extract(TSX, "v.tsx", src, "sha")] == ["View"]


def test_spec_for_extensions():
    assert spec_for("a/b.go") is GO
    assert spec_for("a.ts") is TYPESCRIPT
    assert spec_for("a.tsx") is TSX
    assert spec_for("a.py") is None


def test_norm_hash_ignores_names_literals_and_comments():
    renamed = (
        GO_SRC.replace(b"Sum", b"Total")
        .replace(b"total", b"acc")
        .replace(b"x < 0", b"x < 10")
        .replace(b"total += 0\n", b"acc += 0 // nada\n")
    )
    a = _by_name(extract(GO, "x.go", GO_SRC, "sha"))["Sum"]
    b = _by_name(extract(GO, "y.go", renamed, "sha"))["Total"]
    assert a.content_hash != b.content_hash
    assert a.norm_hash == b.norm_hash
    assert a.shingles == b.shingles

    # Un cambio estructural (otro operador) sí cambia la huella.
    changed = _by_name(extract(GO, "z.go", GO_SRC.replace(b"x < 0", b"x > 0"), "sha"))["Sum"]
    assert changed.norm_hash != a.norm_hash


def test_norm_hash_typescript():
    src = TS_SRC.replace(b"add(a: number, b: number)", b"plus(p: number, q: number)")
    src = src.replace(b"const r = a + b;\n  return r;", b'const s = p + q; // suma\n  return s;')
    a = _by_name(extract(TYPESCRIPT, "a.ts", TS_SRC, "sha"))["add"]
    b = _by_name(extract(TYPESCRIPT, "b.ts", src, "sha"))["plus"]
    assert a.content_hash != b.content_hash
    assert a.norm_hash == b.norm_hash
