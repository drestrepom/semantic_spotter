import pytest

from semantic_spotter.filters import PathFilter, parse_ignore_file


@pytest.mark.parametrize(
    ("patterns", "path", "excluded"),
    [
        # `*` es un segmento; `**` cualquier cantidad (incluso cero).
        (["apps/*/migrations/"], "apps/billing-api/migrations/2026_x/main.go", True),
        (["apps/*/migrations/"], "apps/billing-api/internal/migrations.go", False),
        (["apps/*/migrations"], "apps/a/migrations/x/main.go", True),
        (["apps/*/migrations"], "apps/a/b/migrations/main.go", False),
        (["apps/**/migrations"], "apps/a/b/migrations/main.go", True),
        # Sin `/` matchea a cualquier profundidad, como en .gitignore.
        (["migrations"], "apps/a/migrations/main.go", True),
        (["*.gen.go"], "pkg/deep/x.gen.go", True),
        (["*.gen.go"], "x.gen.go", True),
        (["*.gen.go"], "pkg/x.go", False),
        # `/` inicial ancla a la raíz.
        (["/scripts"], "scripts/a.go", True),
        (["/scripts"], "apps/scripts/a.go", False),
        # `/` final: solo directorios, no un archivo con ese nombre.
        (["fixtures/"], "pkg/fixtures", False),
        (["fixtures/"], "pkg/fixtures/a.ts", True),
        # `!` re-incluye; gana el último patrón que matchea.
        (["apps/*/migrations/", "!apps/core-api/migrations/"], "apps/core-api/migrations/m.go", False),
        (["apps/*/migrations/", "!apps/core-api/migrations/"], "apps/x-api/migrations/m.go", True),
        (["**/components/ui/**"], "apps/web/src/components/ui/button.tsx", True),
        ([], "anything.go", False),
    ],
)
def test_path_filter(patterns, path, excluded):
    assert PathFilter(patterns).excluded(path) is excluded


def test_parse_ignore_file_skips_comments_and_blanks():
    content = "# migraciones de un solo uso\napps/*/migrations/\n\n  \n*.gen.go\n"
    assert parse_ignore_file(content) == ["apps/*/migrations/", "*.gen.go"]
    assert not PathFilter(["# solo comentario", ""])
