from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

IGNORE_FILENAME = ".semantic-spotterignore"


@dataclass(frozen=True, slots=True)
class _Rule:
    glob: str
    negated: bool
    dir_only: bool


def _compile(pattern: str) -> _Rule | None:
    """Traduce un patrón estilo `.gitignore` a un glob para `PurePath.full_match`.

    - Sin `/` (salvo uno final) matchea a cualquier profundidad: `*.gen.go`.
    - Con `/` al inicio o en medio queda anclado a la raíz: `apps/*/migrations`.
    - Con `/` final solo aplica a directorios: `fixtures/`.
    - `!` al inicio re-incluye lo que un patrón anterior excluyó.
    """
    pattern = pattern.strip()
    if not pattern or pattern.startswith("#"):
        return None
    negated = pattern.startswith("!")
    pattern = pattern.removeprefix("!")
    dir_only = pattern.endswith("/")
    core = pattern.rstrip("/")
    anchored = "/" in core
    core = core.lstrip("/")
    if not core:
        return None
    if not anchored:
        core = "**/" + core
    return _Rule(core, negated, dir_only)


class PathFilter:
    """Decide si una ruta queda excluida; gana el último patrón que matchea."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self._rules = [rule for p in patterns if (rule := _compile(p)) is not None]
        # Hay miles de snippets por ruta: cada ruta se evalúa una sola vez.
        self._cache: dict[str, bool] = {}

    def __bool__(self) -> bool:
        return bool(self._rules)

    def excluded(self, path: str) -> bool:
        if (cached := self._cache.get(path)) is not None:
            return cached
        result = self._cache[path] = self._match(path)
        return result

    def _match(self, path: str) -> bool:
        target = PurePosixPath(path)
        # Un patrón que nombra un directorio excluye todo lo que hay debajo.
        directories = [d for d in target.parents if d != PurePosixPath(".")]
        result = False
        for rule in self._rules:
            candidates = directories if rule.dir_only else [target, *directories]
            if any(c.full_match(rule.glob) for c in candidates):
                result = not rule.negated
        return result


def parse_ignore_file(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.strip() and not line.startswith("#")]
