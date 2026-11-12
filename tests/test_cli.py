import json
import re

import pytest
from typer.testing import CliRunner

from conftest import git

from semantic_spotter import cli

FUNC = """package {pkg}

func {name}(items []string) map[string]int {{
	counts := map[string]int{{}}
	for _, it := range items {{
		counts[it]++
	}}
	return counts
}}
"""

OTHER = """package c

func Render(w Writer, title string) error {
	header := "<h1>" + title + "</h1>"
	_, err := w.Write([]byte(header))
	return err
}
"""


@pytest.fixture
def repo(make_repo, fake_embedder, monkeypatch):
    monkeypatch.setattr(cli, "make_embedder", lambda model: fake_embedder)
    monkeypatch.setattr(cli, "DIM", fake_embedder.dim)
    path, _ = make_repo(
        {
            "a/a.go": FUNC.format(pkg="a", name="CountItems"),
            "b/b.go": FUNC.format(pkg="b", name="Tally"),
            "c/c.go": OTHER,
            "mig/x/main.go": FUNC.format(pkg="main", name="Count"),
        }
    )
    return path


def test_scan_reports_clone_group(repo):
    result = CliRunner().invoke(cli.app, ["scan", str(repo), "--json", "--threshold", "0.8"])
    assert result.exit_code == 0, result.output
    groups = json.loads(result.stdout)
    assert len(groups) == 1
    assert {m["file_path"] for m in groups[0]["members"]} == {"a/a.go", "b/b.go", "mig/x/main.go"}
    # Solo el par de funciones: el par de `for` internos queda subsumido.
    assert all(m["unit"] == "function" for m in groups[0]["members"])
    # Solo difieren en nombres: renombrado (T2), y se borrarían dos de las tres copias.
    assert groups[0]["clone_type"] == 2
    assert groups[0]["deletable_lines"] == 14
    assert not groups[0]["blocks_only"] and not groups[0]["interface_like"]


def test_scan_rich_report_sections(repo):
    result = CliRunner().invoke(cli.app, ["scan", str(repo), "--threshold", "0.8"])
    assert result.exit_code == 0, result.output
    assert "T2 · Renombrados" in result.stdout
    assert "líneas eliminables en grupos accionables" in result.stderr


def test_hidden_sections_need_show_all(make_repo, fake_embedder, monkeypatch):
    monkeypatch.setattr(cli, "make_embedder", lambda model: fake_embedder)
    monkeypatch.setattr(cli, "DIM", fake_embedder.dim)
    method = """package {pkg}

type impl struct{{}}

func (impl) Price(sku string) int {{
	if sku == "{lit}" {{
		return {n}
	}}
	return len(sku) * {n}
}}
"""
    repo, _ = make_repo(
        {
            "x/x.go": method.format(pkg="x", lit="a", n=1),
            "y/y.go": method.format(pkg="y", lit="b", n=2).replace("len(sku)", "len(sku)+1"),
        }
    )
    args = ["scan", str(repo), "--threshold", "0.5", "--t3-jaccard", "1.0"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "Probable interfaz" in result.stdout and "Ocultos" in result.stdout
    assert "impl.Price" not in result.stdout
    result = CliRunner().invoke(cli.app, [*args, "--show-all"])
    assert "impl.Price" in result.stdout


def test_dry_run_does_not_embed(repo, fake_embedder):
    result = CliRunner().invoke(cli.app, ["scan", str(repo), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert fake_embedder.embedded == []
    assert "sin embedding" in result.stderr


def test_top_mode(repo):
    result = CliRunner().invoke(cli.app, ["scan", str(repo), "--json", "--top", "2"])
    assert result.exit_code == 0, result.output
    pairs = json.loads(result.stdout)
    assert len(pairs) == 2
    assert pairs[0]["score"] >= pairs[1]["score"]
    assert all(1 <= p["clone_type"] <= 4 for p in pairs)


def test_exclude_glob_from_cli_and_ignore_file(repo, make_repo):
    args = ["scan", str(repo), "--json", "--threshold", "0.8"]
    result = CliRunner().invoke(cli.app, [*args, "--exclude", "mig/"])
    assert result.exit_code == 0, result.output
    groups = json.loads(result.stdout)
    assert {m["file_path"] for m in groups[0]["members"]} == {"a/a.go", "b/b.go"}
    assert "excluidos por patrones" in result.stderr

    # El archivo versionado en el commit aplica sin pasar flags.
    make_repo({".semantic-spotterignore": "# ruido\nmig/\nb/\n"})
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


# --- check ---------------------------------------------------------------------

LONG = """package {pkg}

func {name}(orders []Order, limit int) []string {{
	seen := map[string]bool{{}}
	out := make([]string, 0, limit)
	for _, o := range orders {{
		if o.Status != "{status}" || seen[o.ID] {{
			continue
		}}
		seen[o.ID] = true
		out = append(out, o.ID)
		if len(out) >= limit {{
			break
		}}
	}}
	return out
}}
"""

TWEAKED = LONG.replace("seen[o.ID] = true", "seen[o.ID] = true\n\t\tlog.Println(o.ID)")


@pytest.fixture
def check_repo(make_repo, fake_embedder, monkeypatch):
    """Repo con `main` (dos archivos) y una rama `feature` sin cambios aún."""
    monkeypatch.setattr(cli, "make_embedder", lambda model: fake_embedder)
    monkeypatch.setattr(cli, "DIM", fake_embedder.dim)
    monkeypatch.setattr(cli, "_api_key", lambda: "test")
    path, _ = make_repo(
        {"a/a.go": LONG.format(pkg="a", name="PendingIDs", status="pending"), "c/c.go": OTHER}
    )
    git(path, "branch", "-M", "main")
    git(path, "checkout", "-q", "-b", "feature")
    return path


def _check(repo, *args):
    return CliRunner().invoke(cli.app, ["check", str(repo), "--json", *args])


def _plain(text):
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


def test_check_flags_renamed_copy(check_repo, make_repo):
    make_repo({"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 1, result.output
    [group] = json.loads(result.stdout)
    assert group["clone_type"] == 2
    changed = {m["file_path"]: m["changed"] for m in group["members"]}
    assert changed == {"a/a.go": False, "b/b.go": True}
    # La función y su `for` interno; el par de bloques queda subsumido en el de funciones.
    assert "2 snippets cambiados" in _plain(result.stderr)


def test_check_ignores_moves_and_line_shifts(check_repo, make_repo):
    (check_repo / "c/c.go").rename(check_repo / "c/render.go")
    shifted = LONG.format(pkg="a", name="PendingIDs", status="pending").replace(
        "package a\n", "package a\n\n// cabecera\n"
    )
    make_repo({"a/a.go": shifted})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert "0 snippets cambiados" in _plain(result.stderr)


def test_check_staged_sees_index(check_repo):
    (check_repo / "d").mkdir()
    (check_repo / "d/d.go").write_text(LONG.format(pkg="a", name="PendingIDs", status="pending"))
    assert _check(check_repo, "--staged", "--structural").exit_code == 0
    git(check_repo, "add", "d/d.go")
    result = _check(check_repo, "--staged", "--structural")
    assert result.exit_code == 1, result.output
    [group] = json.loads(result.stdout)
    assert group["clone_type"] == 1


def test_check_fail_on_and_semantic_warnings(check_repo, make_repo, fake_embedder):
    make_repo({"b/b.go": TWEAKED.format(pkg="b", name="PendingIDs", status="pending")})
    # Casi idéntico (T3): falla por defecto, no con --fail-on t1.
    result = _check(check_repo, "--structural", "--t3-jaccard", "0.6")
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)[0]["clone_type"] == 3
    assert _check(check_repo, "--structural", "--t3-jaccard", "0.6", "--fail-on", "t1").exit_code == 0

    # Con embeddings, el mismo par es T4 si se exige más Jaccard: solo avisa.
    result = _check(check_repo, "--t3-jaccard", "1.0", "--threshold", "0.8")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["clone_type"] == 4
    assert fake_embedder.embedded


def test_check_without_api_key_skips_t4(check_repo, make_repo, monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_api_key", lambda: None)
    make_repo({"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")})
    result = _check(check_repo)
    assert result.exit_code == 1, result.output
    assert "se omiten los clones semánticos" in result.stderr
    assert fake_embedder.embedded == []


def test_index_then_check_shows_first_seen(check_repo, make_repo):
    result = CliRunner().invoke(cli.app, ["index", str(check_repo)])
    assert result.exit_code == 0, result.output
    assert "1 commits recorridos" in _plain(result.stderr)
    original = git(check_repo, "rev-parse", "HEAD")

    make_repo({"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")})
    [group] = json.loads(_check(check_repo, "--structural").stdout)
    first_seen = {m["file_path"]: m["first_seen"] for m in group["members"]}
    assert first_seen["a/a.go"]["commit"] == original
    assert first_seen["b/b.go"] is None  # el commit nuevo todavía no se recorrió
