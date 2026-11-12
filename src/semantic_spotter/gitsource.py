import asyncio
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_NOISE_DIRS = frozenset({"vendor", "node_modules"})
_GENERATED_SUFFIXES = (".d.ts", ".pb.go")
_TEST_PATTERNS = re.compile(r"(_test\.go|\.(test|spec)\.[cm]?tsx?)$")
# Convención de Go (y de muchos generadores TS): https://go.dev/s/generatedcode
_GENERATED_HEADER = re.compile(rb"^// Code generated .* DO NOT EDIT\.$", re.MULTILINE)


class GitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GitFile:
    path: str
    oid: str


@dataclass(frozen=True, slots=True)
class HistoryCommit:
    sha: str
    committed_at: int  # epoch, segundos
    # archivos que el commit agregó o modificó respecto de su primer padre
    changed: list[GitFile]


async def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repo),
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(stdin)
    if proc.returncode != 0:
        raise GitError(stderr.decode("utf-8", "replace").strip())
    return stdout


async def repo_root(path: Path) -> Path:
    return Path((await _git(path, "rev-parse", "--show-toplevel")).decode().strip())


async def common_dir(repo: Path) -> Path:
    """Directorio común de git: el mismo para todos los worktrees del repo."""
    out = (await _git(repo, "rev-parse", "--git-common-dir")).decode().strip()
    # Puede venir relativo (p. ej. `.git`) respecto del repo.
    return (repo / out).resolve()


async def resolve_rev(repo: Path, rev: str) -> str:
    return (await _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")).decode().strip()


async def head_commit(repo: Path) -> str | None:
    """Commit de HEAD, o None si el repo (ya validado) todavía no tiene commits."""
    try:
        return await resolve_rev(repo, "HEAD")
    except GitError:
        return None


async def empty_tree(repo: Path) -> str:
    """Oid del árbol vacío en el formato de hash del repo (SHA-1 o SHA-256)."""
    return (await _git(repo, "hash-object", "-t", "tree", "--stdin", stdin=b"")).decode().strip()


async def tree_of(repo: Path, rev: str) -> str:
    return (await _git(repo, "rev-parse", "--verify", f"{rev}^{{tree}}")).decode().strip()


async def staged_tree(repo: Path) -> str:
    """Árbol del index (lo stageado); `write-tree` solo agrega objetos, no toca refs."""
    return (await _git(repo, "write-tree")).decode().strip()


async def merge_base(repo: Path, a: str, b: str) -> str:
    return (await _git(repo, "merge-base", a, b)).decode().strip()


async def list_files(repo: Path, tree_ish: str) -> list[GitFile]:
    """Blobs regulares del árbol (o commit) dado, sin symlinks ni submódulos."""
    out = await _git(repo, "ls-tree", "-r", "-z", tree_ish)
    files = []
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, oid = meta.split(b" ")
        if kind == b"blob" and mode != b"120000":
            files.append(GitFile(path=path.decode("utf-8", "replace"), oid=oid.decode()))
    return files


_NULL_OID = "0" * 40


async def history(repo: Path, rev: str, since: str | None = None) -> list[HistoryCommit]:
    """Commits de `since..rev` (o de todo `rev`), padres antes que hijos.

    Un merge se compara contra su primer padre; los commits de la rama
    integrada aparecen antes, así que lo que traen se les atribuye a ellos.
    """
    out = await _git(
        repo,
        "-c",
        "core.quotePath=false",
        "log",
        "--reverse",
        "--topo-order",
        "--no-renames",
        "--raw",
        "--no-abbrev",
        "--root",
        "--diff-merges=first-parent",
        "--format=commit %H %ct",
        f"{since}..{rev}" if since else rev,
    )
    commits: list[HistoryCommit] = []
    for line in out.decode("utf-8", "replace").splitlines():
        if line.startswith("commit "):
            _, sha, ts = line.split(" ")
            commits.append(HistoryCommit(sha, int(ts), []))
        elif line.startswith(":") and commits:
            meta, path = line.split("\t", 1)
            _, new_mode, _, new_oid, status = meta[1:].split(" ")
            if status != "D" and new_oid != _NULL_OID and new_mode.startswith("100"):
                commits[-1].changed.append(GitFile(path=path, oid=new_oid))
    return commits


async def read_blobs(repo: Path, oids: list[str]) -> dict[str, bytes]:
    """Lee muchos blobs con un único proceso `git cat-file --batch`."""
    if not oids:
        return {}
    out = await _git(repo, "cat-file", "--batch", stdin="".join(f"{o}\n" for o in oids).encode())
    blobs: dict[str, bytes] = {}
    pos = 0
    while pos < len(out):
        header_end = out.index(b"\n", pos)
        header = out[pos:header_end].split(b" ")
        pos = header_end + 1
        if len(header) == 2 and header[1] == b"missing":
            continue
        oid, size = header[0].decode(), int(header[2])
        blobs[oid] = out[pos : pos + size]
        pos += size + 1  # contenido + "\n"
    return blobs


async def read_file(repo: Path, tree_ish: str, path: str) -> bytes | None:
    """Contenido de `path` en el árbol (o commit), o None si no existe ahí."""
    try:
        return await _git(repo, "cat-file", "blob", f"{tree_ish}:{path}")
    except GitError:
        return None


def is_candidate(path: str, *, include_tests: bool) -> bool:
    if path.endswith(_GENERATED_SUFFIXES):
        return False
    if _NOISE_DIRS.intersection(PurePosixPath(path).parts[:-1]):
        return False
    if not include_tests and _TEST_PATTERNS.search(path):
        return False
    return True


def is_generated(source: bytes) -> bool:
    return _GENERATED_HEADER.search(source) is not None
