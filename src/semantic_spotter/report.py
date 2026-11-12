from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from semantic_spotter.detect import CLONE_TYPES, Group, Pair
from semantic_spotter.models import Snippet
from semantic_spotter.store import Sighting

# (ruta, content_hash) → primer avistamiento, de `semantic-spotter index`.
Seen = Mapping[tuple[str, str], Sighting]


def _date(sighting: Sighting) -> str:
    return datetime.fromtimestamp(sighting.committed_at, UTC).date().isoformat()


def _first_seen(s: Snippet, seen: Seen) -> dict[str, Any]:
    sighting = seen.get((s.file_path, s.content_hash))
    if sighting is None:
        return {"first_seen": None}
    return {"first_seen": {"commit": sighting.commit_sha, "date": _date(sighting)}}


def _member(s: Snippet) -> dict[str, Any]:
    return {
        "file_path": s.file_path,
        "start_line": s.start_line,
        "end_line": s.end_line,
        "unit": s.unit,
        "kind": s.kind,
        "name": s.name,
        "content_hash": s.content_hash,
    }


def groups_to_json(
    snippets: Sequence[Snippet],
    groups: Sequence[Group],
    changed: Collection[int] = (),
    seen: Seen | None = None,
) -> list[dict[str, Any]]:
    """`changed`: índices introducidos por el cambio analizado en `check`."""
    return [
        {
            "size": len(g.members),
            "mean_similarity": round(g.mean_score, 4),
            "generic": g.generic,
            "clone_type": g.clone_type,
            "deletable_lines": g.deletable_lines,
            "blocks_only": g.blocks_only,
            "interface_like": g.interface_like,
            "members": [
                _member(snippets[i])
                | {"max_similarity": round(g.best_score[i], 4)}
                | ({"changed": i in changed} if changed else {})
                | (_first_seen(snippets[i], seen) if seen else {})
                for i in g.members
            ],
            "pairs": [
                {
                    "a": snippets[p.a].location,
                    "b": snippets[p.b].location,
                    "score": round(p.score, 4),
                    "clone_type": p.clone_type,
                }
                for p in g.pairs
            ],
        }
        for g in groups
    ]


def pairs_to_json(snippets: Sequence[Snippet], pairs: Sequence[Pair]) -> list[dict[str, Any]]:
    return [
        {
            "score": round(p.score, 4),
            "clone_type": p.clone_type,
            "a": _member(snippets[p.a]),
            "b": _member(snippets[p.b]),
        }
        for p in pairs
    ]


def print_groups(
    console: Console,
    snippets: Sequence[Snippet],
    groups: Sequence[Group],
    *,
    show_all: bool,
    changed: Collection[int] = (),
    seen: Seen | None = None,
) -> None:
    if not groups:
        console.print("[green]No se encontraron clones por encima del umbral.[/]")
        return
    sections = [
        (f"T{t} · {label}", [g for g in groups if g.actionable and g.clone_type == t])
        for t, label in CLONE_TYPES.items()
    ]
    sections.append(
        ("Probable interfaz / solo bloques", [g for g in groups if not g.actionable])
    )
    start = 1
    for title, section in sections:
        if not section:
            continue
        lines = sum(g.deletable_lines for g in section)
        console.rule(f"{title} · {len(section)} grupos · {lines} líneas eliminables")
        if section[0].actionable or show_all:
            _print_group_tables(
                console, snippets, section, start=start, changed=changed, seen=seen
            )
        else:
            console.print("[dim]Ocultos; usa --show-all para verlos.[/]")
        start += len(section)


def _print_group_tables(
    console: Console,
    snippets: Sequence[Snippet],
    groups: Sequence[Group],
    *,
    start: int,
    changed: Collection[int],
    seen: Seen | None,
) -> None:
    for n, g in enumerate(groups, start):
        title = (
            f"Grupo #{n} · T{g.clone_type} · {len(g.members)} miembros · "
            f"{g.deletable_lines} líneas eliminables · similitud media {g.mean_score:.3f}"
        )
        if g.interface_like:
            title += " · [cyan]métodos homónimos (¿interfaz?)[/]"
        if g.generic:
            title += " · [yellow]genérico (posible patrón/boilerplate)[/]"
        table = Table(title=title, title_justify="left", expand=True)
        table.add_column("Ubicación", overflow="fold")
        table.add_column("Tipo")
        table.add_column("Nombre", overflow="fold")
        table.add_column("Sim. máx.", justify="right")
        if seen:
            table.add_column("Desde")
        for i in sorted(g.members, key=lambda i: -g.best_score[i]):
            s = snippets[i]
            location = f"[bold green]● nuevo[/] {s.location}" if i in changed else s.location
            row = [location, s.kind, s.name, f"{g.best_score[i]:.3f}"]
            if seen:
                sighting = seen.get((s.file_path, s.content_hash))
                row.append(f"{sighting.commit_sha[:9]} {_date(sighting)}" if sighting else "—")
            table.add_row(*row)
        console.print(table)


def print_pairs(console: Console, snippets: Sequence[Snippet], pairs: Sequence[Pair]) -> None:
    table = Table(title="Pares más similares", title_justify="left", expand=True)
    table.add_column("#", justify="right")
    table.add_column("Sim.", justify="right")
    table.add_column("Tipo")
    table.add_column("A", overflow="fold")
    table.add_column("B", overflow="fold")
    for n, p in enumerate(pairs, 1):
        a, b = snippets[p.a], snippets[p.b]
        table.add_row(
            str(n), f"{p.score:.3f}", f"T{p.clone_type}", f"{a.location}\n{a.name}", f"{b.location}\n{b.name}"
        )
    console.print(table)
