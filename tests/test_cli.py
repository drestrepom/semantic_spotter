import json
import os
import re
import subprocess
import sys

import pytest
import voyageai
from typer.testing import CliRunner

from conftest import count_extractions, git

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


@pytest.mark.parametrize(
    "failure",
    [
        "confirm",  # sin TTY, la confirmación de tokens aborta
        voyageai.error.RateLimitError("cuota agotada"),
        RuntimeError("bug"),
    ],
    ids=["confirm-aborta", "voyage", "inesperado"],
)
def test_check_tool_errors_exit_2(check_repo, make_repo, fake_embedder, monkeypatch, failure):
    make_repo({"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")})
    args = []
    if failure == "confirm":
        args = ["--max-tokens-without-confirm", "0"]
    else:

        async def embed(texts):
            raise failure

        monkeypatch.setattr(fake_embedder, "embed", embed)
    result = _check(check_repo, *args)
    # 1 queda solo para grupos que fallan por --fail-on.
    assert result.exit_code == 2, result.output


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


# --- marca de ignorar ------------------------------------------------------------


def _marked(source, marker="semantic-spotter:ignore -- adaptador temporal"):
    """Pone `// <marker>` en la línea anterior a cada función del archivo."""
    return source.replace("\nfunc ", f"\n// {marker}\nfunc ")


def test_check_passes_marked_copy(check_repo, make_repo):
    make_repo({"b/b.go": _marked(LONG.format(pkg="b", name="ShippedIDs", status="shipped"))})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert "1 grupos omitidos por marca de ignorar" in _plain(result.stderr)


@pytest.mark.parametrize(
    "marker", ["semantic-spotter:ignore", "semantic-spotter:ignore --", "semantic-spotter:ignore --  "]
)
def test_check_mark_without_reason_does_not_count(check_repo, make_repo, marker):
    copy = _marked(LONG.format(pkg="b", name="ShippedIDs", status="shipped"), marker)
    make_repo({"b/b.go": copy})
    assert _check(check_repo, "--structural").exit_code == 1


def test_check_marking_existing_function_changes_nothing(check_repo, make_repo):
    make_repo({"a/a.go": _marked(LONG.format(pkg="a", name="PendingIDs", status="pending"))})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 0, result.output
    assert "0 snippets cambiados" in _plain(result.stderr)


def test_check_mark_does_not_cover_new_copies(check_repo, make_repo):
    make_repo({"a/a.go": _marked(LONG.format(pkg="a", name="PendingIDs", status="pending"))})
    make_repo({"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 1, result.output
    [group] = json.loads(result.stdout)
    assert {m["file_path"] for m in group["members"]} == {"a/a.go", "b/b.go"}


def test_check_needs_every_changed_member_marked(check_repo, make_repo):
    make_repo(
        {
            "b/b.go": _marked(LONG.format(pkg="b", name="ShippedIDs", status="shipped")),
            "d/d.go": LONG.format(pkg="d", name="DoneIDs", status="done"),
        }
    )
    result = _check(check_repo, "--structural")
    assert result.exit_code == 1, result.output
    [group] = json.loads(result.stdout)
    assert {m["file_path"] for m in group["members"]} == {"a/a.go", "b/b.go", "d/d.go"}


# El `for` de LONG dentro de una función que por lo demás no se le parece.
HOST = """package b

func Summarize(orders []Order, limit int, w io.Writer) error {
	if limit <= 0 {
		return errors.New("limit")
	}
	total := 0
	for _, o := range orders {
		total += o.Amount
	}
	seen := map[string]bool{}
	out := make([]string, 0, limit)
	for _, o := range orders {
		if o.Status != "pending" || seen[o.ID] {
			continue
		}
		seen[o.ID] = true
		out = append(out, o.ID)
		if len(out) >= limit {
			break
		}
	}
	_, err := fmt.Fprintf(w, "%d %v", total, out)
	return err
}
"""


@pytest.mark.parametrize("marked", [False, True])
def test_check_mark_covers_inner_blocks(check_repo, make_repo, marked):
    make_repo({"b/b.go": _marked(HOST) if marked else HOST})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 0, result.output
    groups = json.loads(result.stdout)
    if marked:
        assert groups == []
    else:
        # Sin marca, el `for` copiado se reporta como grupo de solo bloques.
        [group] = groups
        assert group["blocks_only"]
        assert {m["file_path"] for m in group["members"]} == {"a/a.go", "b/b.go"}


def test_check_mark_silences_t3_warning(check_repo, make_repo):
    make_repo({"b/b.go": _marked(TWEAKED.format(pkg="b", name="PendingIDs", status="pending"))})
    result = _check(check_repo, "--structural", "--t3-jaccard", "0.6", "--fail-on", "t1")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


TS_BODY = """
  const seen = new Set<string>();
  const out: string[] = [];
  for (const o of orders) {
    if (o.status !== "pending" || seen.has(o.id)) {
      continue;
    }
    seen.add(o.id);
    out.push(o.id);
    if (out.length >= limit) {
      break;
    }
  }
  return out;
"""
TS_MARK = "// semantic-spotter:ignore -- variantes por tipo de orden\n"


@pytest.mark.parametrize("marked", [False, True])
def test_check_mark_in_typescript(check_repo, make_repo, marked):
    mark = TS_MARK if marked else ""
    signature = "(orders: Order[], limit: number): string[]"
    make_repo(
        {
            "t/a.ts": f"{mark}export function pendingIds{signature} {{{TS_BODY}}}\n",
            "t/b.ts": f"{mark}export const shippedIds = {signature} => {{{TS_BODY}}};\n",
            "t/c.ts": f"export class Repo {{\n  {mark}  ids{signature} {{{TS_BODY}}}\n}}\n",
        }
    )
    result = _check(check_repo, "--structural")
    assert result.exit_code == (0 if marked else 1), result.output


def test_check_failure_explains_both_ways_out(check_repo, make_repo):
    make_repo({"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")})
    result = _check(check_repo, "--structural")
    assert result.exit_code == 1, result.output
    stderr = " ".join(_plain(result.stderr).split())
    assert "extrae la lógica compartida" in stderr
    assert "// semantic-spotter:ignore -- <motivo>" in stderr


def test_scan_omits_groups_with_every_member_marked(repo, make_repo):
    make_repo(
        {
            "a/a.go": _marked(FUNC.format(pkg="a", name="CountItems")),
            "b/b.go": _marked(FUNC.format(pkg="b", name="Tally")),
        }
    )
    args = ["scan", str(repo), "--json", "--threshold", "0.8"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    [group] = json.loads(result.stdout)
    assert len(group["members"]) == 3

    make_repo({"mig/x/main.go": _marked(FUNC.format(pkg="main", name="Count"))})
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


# --- índice compartido -----------------------------------------------------------


def test_worktree_shares_index_in_git_common_dir(check_repo, tmp_path, monkeypatch):
    assert _check(check_repo, "--structural").exit_code == 0
    assert (check_repo / ".git" / "semantic-spotter").is_dir()
    assert not (check_repo / ".semantic-spotter").exists()

    worktree = tmp_path / "wt"
    git(check_repo, "worktree", "add", "-q", "-b", "otra", str(worktree), "main")
    (worktree / "b").mkdir()
    (worktree / "b/b.go").write_text(LONG.format(pkg="b", name="ShippedIDs", status="shipped"))
    git(worktree, "add", "-A")
    git(worktree, "commit", "-q", "-m", "copia")

    extracted = count_extractions(monkeypatch)
    result = _check(worktree, "--structural")
    assert result.exit_code == 1, result.output
    # a/a.go y c/c.go ya los extrajo el otro worktree: solo se extrae lo nuevo.
    assert extracted == ["b/b.go"]
    assert not (worktree / ".semantic-spotter").exists()


def test_index_path_from_env(check_repo, tmp_path, monkeypatch):
    db = tmp_path / "indice" / "pruebas"
    monkeypatch.setenv("SEMANTIC_SPOTTER_DB", str(db))
    assert _check(check_repo, "--structural").exit_code == 0
    assert any(db.iterdir())
    assert not (check_repo / ".git" / "semantic-spotter").exists()

    # Otro índice vacío: lo ya extraído en el de por defecto no cuenta.
    monkeypatch.delenv("SEMANTIC_SPOTTER_DB")
    extracted = count_extractions(monkeypatch)
    assert _check(check_repo, "--structural").exit_code == 0
    assert sorted(extracted) == ["a/a.go", "c/c.go"]


def test_parallel_checks_share_index(check_repo, make_repo):
    # Varios archivos para que las dos corridas compitan por las mismas escrituras.
    make_repo(
        {"b/b.go": LONG.format(pkg="b", name="ShippedIDs", status="shipped")}
        | {f"o{i}/o.go": OTHER.replace("Render", f"Render{i}") for i in range(8)}
    )
    command = [
        sys.executable,
        "-c",
        "from semantic_spotter import main; main()",
        "check",
        str(check_repo),
        "--json",
        "--structural",
    ]
    env = {k: v for k, v in os.environ.items() if k != "SEMANTIC_SPOTTER_DB"}
    procs = [
        subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    outputs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [1, 1], [err for _, err in outputs]

    # Mismo resultado que una corrida en serie sobre el índice ya escrito.
    serial = _check(check_repo, "--structural")
    assert serial.exit_code == 1, serial.output
    expected = json.loads(serial.stdout)
    assert [m["file_path"] for m in expected[0]["members"]] == ["a/a.go", "b/b.go"]
    for out, _ in outputs:
        assert json.loads(out) == expected
