import asyncio
import json
import os
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
import voyageai
from dotenv import find_dotenv, load_dotenv
from rich.console import Console

from semantic_spotter import gitsource
from semantic_spotter.detect import (
    Group,
    candidate_pairs,
    changed_occurrences,
    classify_pairs,
    drop_ignored,
    group_pairs,
    merge_pairs,
    query_pairs,
    renamed_pairs,
    structural_pairs,
    suppress_subsumed,
)
from semantic_spotter.embedder import Embedder, VoyageEmbedder
from semantic_spotter.filters import IGNORE_FILENAME
from semantic_spotter.models import Snippet
from semantic_spotter.pipeline import (
    ExtractOptions,
    RateLimits,
    apply_path_filter,
    index_commit,
    index_history,
    index_tree,
    load_path_filter,
    plan_embeddings,
    run_embeddings,
)
from semantic_spotter.report import groups_to_json, pairs_to_json, print_groups, print_pairs
from semantic_spotter.store import IndexBusy, Store

DEFAULT_MODEL = "voyage-code-4"
DIM = 1024
_PLACEHOLDER_KEY = "pa-REPLACE_ME"

app = typer.Typer(no_args_is_help=True, add_completion=False)
# Progreso y avisos a stderr para que `--json` deje stdout limpio.
err = Console(stderr=True)

# --- opciones compartidas por `scan` y `check` ---------------------------------

PathArg = Annotated[Path, typer.Argument(help="Ruta dentro del repo git a analizar.")]
ModelOpt = Annotated[str, typer.Option(help="Modelo de embeddings de Voyage.")]
ThresholdOpt = Annotated[float, typer.Option(help="Similitud coseno mínima.")]
ExcludeOpt = Annotated[
    list[str] | None,
    typer.Option(
        help="Glob estilo .gitignore a excluir (repetible), p. ej. 'apps/*/migrations/'. "
        f"Se suma a los patrones de {IGNORE_FILENAME} en el árbol analizado."
    ),
]
T3JaccardOpt = Annotated[
    float,
    typer.Option(help="Jaccard mínimo de k-gramas normalizados para clasificar un par como T3."),
]
ShowAllOpt = Annotated[
    bool, typer.Option(help="Mostrar también los grupos de probable interfaz / solo bloques.")
]
IncludeTestsOpt = Annotated[bool, typer.Option(help="Incluir archivos de test.")]
MinFuncLinesOpt = Annotated[int, typer.Option(help="Líneas no vacías mínimas por función.")]
MinBlockLinesOpt = Annotated[int, typer.Option(help="Líneas no vacías mínimas por bloque.")]
MaxGroupOpt = Annotated[int, typer.Option(help="Tamaño a partir del cual un grupo es 'genérico'.")]
MaxTokensOpt = Annotated[
    int, typer.Option(help="Pedir confirmación si hay que embeber más tokens que esto.")
]
ConcurrencyOpt = Annotated[int, typer.Option(help="Requests a Voyage en paralelo.")]
RpmOpt = Annotated[
    int | None, typer.Option(help="Requests por minuto permitidas por tu cuenta de Voyage.")
]
TpmOpt = Annotated[
    int | None, typer.Option(help="Tokens por minuto permitidos por tu cuenta de Voyage.")
]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="No pedir confirmación.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Imprimir el resultado como JSON.")]


@dataclass(frozen=True)
class EmbedSettings:
    model: str
    max_tokens_without_confirm: int
    concurrency: int
    limits: RateLimits
    yes: bool


def _api_key() -> str | None:
    key = os.environ.get("VOYAGE_API_KEY")
    return key if key and key != _PLACEHOLDER_KEY else None


def make_embedder(model: str) -> Embedder:
    api_key = _api_key()
    if api_key is None:
        err.print("[red]Falta VOYAGE_API_KEY: complétala en .env o en el entorno.[/]")
        raise typer.Exit(2)
    return VoyageEmbedder(model, DIM, api_key=api_key)


async def _open_store(repo: Path, model: str) -> Store:
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(repo / ".env")
    return await Store.open(
        repo,
        model,
        DIM,
        on_lock_wait=lambda: err.print(
            "Esperando a que otro proceso termine de escribir el índice…"
        ),
    )


def _git_error(exc: gitsource.GitError) -> typer.Exit:
    err.print(f"[red]Error de git:[/] {exc}")
    return typer.Exit(2)


def _run[T](command: Coroutine[Any, Any, T]) -> T:
    """Corre el comando; un error de la herramienta sale con 2, nunca con 1.

    El 1 queda para `check` cuando encuentra clones de `--fail-on`, así quien
    lo llama (el hook) no confunde un bug con un duplicado.
    """
    try:
        return asyncio.run(command)
    except typer.Exit:
        raise
    except typer.Abort as exc:
        err.print("[red]Cancelado.[/]")
        raise typer.Exit(2) from exc
    except IndexBusy as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
    except Exception as exc:
        err.print_exception()
        err.print(f"[red]Error de semantic-spotter ({type(exc).__name__}):[/] {exc}")
        raise typer.Exit(2) from exc


async def _ensure_embeddings(
    store: Store,
    repo: Path,
    snippets: Sequence[Snippet],
    settings: EmbedSettings,
    *,
    dry_run: bool = False,
    missing: Sequence[str] | None = None,
) -> bool:
    """Embebe los hashes que faltan; False si es `dry_run` y no se embebió."""
    embedder: Embedder | None = None

    def get_embedder() -> Embedder:
        nonlocal embedder
        if embedder is None:
            embedder = make_embedder(settings.model)
        return embedder

    with err.status("Contando tokens de snippets nuevos…"):
        plan = await plan_embeddings(store, repo, snippets, get_embedder, missing)
    err.print(f"{len(plan.hashes)} snippets sin embedding · ~{plan.total_tokens:,} tokens")
    if dry_run:
        return False
    if not plan.hashes:
        return True
    if plan.total_tokens > settings.max_tokens_without_confirm and not settings.yes:
        typer.confirm(f"¿Embeber ~{plan.total_tokens:,} tokens con {settings.model}?", abort=True)
    done = 0

    def progress(n: int) -> None:
        nonlocal done
        done += n
        err.print(f"  embebidos {done}/{len(plan.hashes)}")

    try:
        await run_embeddings(
            store,
            get_embedder(),
            plan,
            on_batch=progress,
            limits=settings.limits,
            concurrency=settings.concurrency,
        )
    except voyageai.error.VoyageError as exc:
        err.print(f"[red]Error de Voyage ({type(exc).__name__}):[/] {exc}")
        err.print(f"Se guardaron {done} embeddings; vuelve a correr para continuar.")
        raise typer.Exit(2) from exc
    return True


def _members(snippets: Sequence[Snippet], groups: Sequence[Group]) -> list[Snippet]:
    return [snippets[i] for g in groups for i in g.members]


def _drop_ignored(
    snippets: Sequence[Snippet], groups: Sequence[Group], changed: set[int] | None = None
) -> list[Group]:
    """`drop_ignored` avisando cuántos grupos omitió por marca de ignorar."""
    kept = drop_ignored(snippets, groups, changed)
    if omitted := len(groups) - len(kept):
        err.print(f"{omitted} grupos omitidos por marca de ignorar")
    return kept


@app.callback()
def _root() -> None:
    """Detecta clones semánticos en repos Go/TypeScript."""


# --- scan ----------------------------------------------------------------------


@app.command()
def scan(
    path: PathArg = Path("."),
    rev: Annotated[str, typer.Option(help="Commit/rama/tag a escanear.")] = "HEAD",
    model: ModelOpt = DEFAULT_MODEL,
    threshold: ThresholdOpt = 0.90,
    top: Annotated[
        int | None, typer.Option(help="Mostrar los N pares más similares, sin umbral.")
    ] = None,
    exclude: ExcludeOpt = None,
    include_exact: Annotated[
        bool, typer.Option(help="En --top, incluir pares con contenido idéntico.")
    ] = False,
    t3_jaccard: T3JaccardOpt = 0.7,
    show_all: ShowAllOpt = False,
    include_tests: IncludeTestsOpt = False,
    min_func_lines: MinFuncLinesOpt = 3,
    min_block_lines: MinBlockLinesOpt = 6,
    max_group: MaxGroupOpt = 8,
    max_tokens_without_confirm: MaxTokensOpt = 2_000_000,
    concurrency: ConcurrencyOpt = 4,
    rpm: RpmOpt = None,
    tpm: TpmOpt = None,
    yes: YesOpt = False,
    dry_run: Annotated[bool, typer.Option(help="Solo estimar tokens, sin llamar a la API.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Busca clones en todo el árbol de un commit."""
    _run(
        _scan(
            path=path,
            rev=rev,
            threshold=threshold,
            top=top,
            exclude=exclude or [],
            include_exact=include_exact,
            t3_jaccard=t3_jaccard,
            show_all=show_all,
            options=ExtractOptions(min_func_lines, min_block_lines, include_tests),
            max_group=max_group,
            embed=EmbedSettings(
                model, max_tokens_without_confirm, concurrency, RateLimits(rpm, tpm), yes
            ),
            dry_run=dry_run,
            as_json=as_json,
        )
    )


async def _scan(
    *,
    path: Path,
    rev: str,
    threshold: float,
    top: int | None,
    exclude: list[str],
    include_exact: bool,
    t3_jaccard: float,
    show_all: bool,
    options: ExtractOptions,
    max_group: int,
    embed: EmbedSettings,
    dry_run: bool,
    as_json: bool,
) -> None:
    try:
        repo = await gitsource.repo_root(path)
        sha = await gitsource.resolve_rev(repo, rev)
    except gitsource.GitError as exc:
        raise _git_error(exc) from exc
    store = await _open_store(repo, embed.model)
    with err.status(f"Extrayendo snippets de {sha[:12]}…"):
        snippets = await index_commit(store, repo, sha, options)
    path_filter = await load_path_filter(repo, sha, exclude)
    kept = apply_path_filter(snippets, path_filter)
    excluded = len(snippets) - len(kept)
    snippets = kept
    err.print(f"{len(snippets)} snippets en {sha[:12]} ({excluded} excluidos por patrones)")

    if not await _ensure_embeddings(store, repo, snippets, embed, dry_run=dry_run):
        return

    matrix = await store.load_vectors([s.content_hash for s in snippets])
    out = Console()
    if top is not None:
        # Pool amplio para que, tras la subsunción, sigan quedando `top` pares.
        pool = candidate_pairs(snippets, matrix, top=top * 10, skip_identical=not include_exact)
        pairs = classify_pairs(
            snippets, suppress_subsumed(snippets, pool)[:top], t3_jaccard=t3_jaccard
        )
        if as_json:
            print(json.dumps(pairs_to_json(snippets, pairs), indent=2, ensure_ascii=False))
        else:
            print_pairs(out, snippets, pairs)
        return

    # Candidatos por coseno más los renombrados (T2) que el embedding no alcanzó.
    pairs = suppress_subsumed(
        snippets,
        merge_pairs(
            candidate_pairs(snippets, matrix, threshold=threshold),
            renamed_pairs(snippets, matrix),
        ),
    )
    pairs = classify_pairs(snippets, pairs, t3_jaccard=t3_jaccard)
    groups = _drop_ignored(snippets, group_pairs(snippets, pairs, max_group=max_group))
    seen = await store.first_seen(_members(snippets, groups), options.key)
    if as_json:
        print(json.dumps(groups_to_json(snippets, groups, seen=seen), indent=2, ensure_ascii=False))
    else:
        print_groups(out, snippets, groups, show_all=show_all, seen=seen)
    deletable = sum(g.deletable_lines for g in groups if g.actionable)
    function_lines = sum(s.lines for s in snippets if s.unit == "function")
    share = deletable / function_lines if function_lines else 0.0
    err.print(
        f"{deletable:,} líneas eliminables en grupos accionables "
        f"({share:.1%} de {function_lines:,} líneas en funciones)"
    )


# --- check ---------------------------------------------------------------------


# Lo lee quien hace commit (a menudo un agente): las dos salidas, sin ir a la doc.
_HOW_TO_FIX = (
    "Para resolverlo, extrae la lógica compartida a una función y reutilízala desde "
    "cada ubicación del grupo. Si la copia es intencional, agrega "
    "`// semantic-spotter:ignore -- <motivo>` en la línea anterior a la función o "
    "bloque nuevo; sin motivo la marca no cuenta."
)


def _parse_fail_on(value: str) -> frozenset[int]:
    types = set()
    for part in filter(None, (p.strip().lower() for p in value.split(","))):
        if part not in ("t1", "t2", "t3", "t4"):
            raise typer.BadParameter(f"tipo desconocido {part!r}; usa t1, t2, t3 o t4")
        types.add(int(part[1]))
    return frozenset(types)


@app.command()
def check(
    path: PathArg = Path("."),
    base: Annotated[
        str | None,
        typer.Option(
            help="Rama/commit base; se compara contra su merge-base con HEAD. "
            "Por defecto 'main', o HEAD con --staged."
        ),
    ] = None,
    staged: Annotated[
        bool, typer.Option(help="Analizar el index (lo stageado) en vez de HEAD, p. ej. en pre-commit.")
    ] = False,
    structural: Annotated[
        bool, typer.Option(help="No usar embeddings: solo T1–T3, sin llamar a Voyage.")
    ] = False,
    fail_on: Annotated[
        str, typer.Option(help="Tipos de clon accionables que hacen fallar (exit 1).")
    ] = "t1,t2,t3",
    model: ModelOpt = DEFAULT_MODEL,
    threshold: ThresholdOpt = 0.92,
    exclude: ExcludeOpt = None,
    t3_jaccard: T3JaccardOpt = 0.7,
    show_all: ShowAllOpt = False,
    include_tests: IncludeTestsOpt = False,
    min_func_lines: MinFuncLinesOpt = 3,
    min_block_lines: MinBlockLinesOpt = 6,
    max_group: MaxGroupOpt = 8,
    max_tokens_without_confirm: MaxTokensOpt = 2_000_000,
    concurrency: ConcurrencyOpt = 4,
    rpm: RpmOpt = None,
    tpm: TpmOpt = None,
    yes: YesOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Busca duplicados solo de los snippets que cambiaron respecto de la base."""
    code = _run(
        _check(
            path=path,
            base=base or ("HEAD" if staged else "main"),
            staged=staged,
            structural=structural,
            fail_on=_parse_fail_on(fail_on),
            threshold=threshold,
            exclude=exclude or [],
            t3_jaccard=t3_jaccard,
            show_all=show_all,
            options=ExtractOptions(min_func_lines, min_block_lines, include_tests),
            max_group=max_group,
            embed=EmbedSettings(
                model, max_tokens_without_confirm, concurrency, RateLimits(rpm, tpm), yes
            ),
            as_json=as_json,
        )
    )
    raise typer.Exit(code)


async def _check(
    *,
    path: Path,
    base: str,
    staged: bool,
    structural: bool,
    fail_on: frozenset[int],
    threshold: float,
    exclude: list[str],
    t3_jaccard: float,
    show_all: bool,
    options: ExtractOptions,
    max_group: int,
    embed: EmbedSettings,
    as_json: bool,
) -> int:
    try:
        repo = await gitsource.repo_root(path)
        head_sha = await gitsource.resolve_rev(repo, "HEAD")
        base_sha = await gitsource.merge_base(repo, base, head_sha)
        head_tree = (
            await gitsource.staged_tree(repo) if staged else await gitsource.tree_of(repo, head_sha)
        )
    except gitsource.GitError as exc:
        raise _git_error(exc) from exc
    store = await _open_store(repo, embed.model)
    head_label = "index" if staged else head_sha[:12]
    with err.status(f"Extrayendo snippets de {base_sha[:12]} y {head_label}…"):
        base_snippets = await index_commit(store, repo, base_sha, options)
        head_snippets = await index_tree(store, repo, head_tree, options)
    path_filter = await load_path_filter(repo, head_tree, exclude)
    base_snippets = apply_path_filter(base_snippets, path_filter)
    snippets = apply_path_filter(head_snippets, path_filter)
    queries = changed_occurrences(base_snippets, snippets)
    err.print(
        f"{len(queries)} snippets cambiados en {head_label} respecto de {base_sha[:12]} "
        f"({len(snippets)} en total)"
    )

    matrix = None
    if queries and not structural:
        missing = await store.missing_hashes([s.content_hash for s in snippets])
        if missing and _api_key() is None:
            err.print("[yellow]Sin VOYAGE_API_KEY: se omiten los clones semánticos (T4).[/]")
        else:
            await _ensure_embeddings(store, repo, snippets, embed, missing=missing)
            matrix = await store.load_vectors([s.content_hash for s in snippets])

    pairs = structural_pairs(snippets, queries, matrix, t3_jaccard=t3_jaccard)
    if matrix is not None:
        pairs = merge_pairs(pairs, query_pairs(snippets, matrix, queries, threshold=threshold))
    pairs = classify_pairs(snippets, suppress_subsumed(snippets, pairs), t3_jaccard=t3_jaccard)
    changed = set(queries)
    groups = _drop_ignored(snippets, group_pairs(snippets, pairs, max_group=max_group), changed)
    seen = await store.first_seen(_members(snippets, groups), options.key)
    if as_json:
        report = groups_to_json(snippets, groups, changed, seen)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_groups(Console(), snippets, groups, show_all=show_all, changed=changed, seen=seen)

    failing = [
        g for g in groups if g.actionable and any(p.clone_type in fail_on for p in g.pairs)
    ]
    labels = ", ".join(f"T{t}" for t in sorted(fail_on)) or "ninguno"
    if failing:
        err.print(f"[red]{len(failing)} grupos con clones {labels} introducidos por el cambio.[/]")
        err.print(_HOW_TO_FIX)
        return 1
    err.print(f"[green]Sin clones {labels} introducidos por el cambio.[/]")
    return 0


# --- index ---------------------------------------------------------------------


@app.command()
def index(
    path: PathArg = Path("."),
    rev: Annotated[str, typer.Option(help="Último commit a recorrer.")] = "HEAD",
    since: Annotated[
        str | None,
        typer.Option(help="Recorrer solo `since..rev`; por defecto, todo el historial de rev."),
    ] = None,
    min_func_lines: MinFuncLinesOpt = 3,
    min_block_lines: MinBlockLinesOpt = 6,
) -> None:
    """Recorre el historial para saber en qué commit apareció cada snippet. No usa Voyage."""
    _run(
        _index(
            path=path,
            rev=rev,
            since=since,
            options=ExtractOptions(min_func_lines, min_block_lines),
        )
    )


async def _index(*, path: Path, rev: str, since: str | None, options: ExtractOptions) -> None:
    try:
        repo = await gitsource.repo_root(path)
        await gitsource.resolve_rev(repo, rev)
        if since is not None:
            await gitsource.resolve_rev(repo, since)
    except gitsource.GitError as exc:
        raise _git_error(exc) from exc
    store = await _open_store(repo, DEFAULT_MODEL)
    with err.status("Leyendo historial…") as status:

        def progress(done: int, total: int) -> None:
            status.update(f"Recorriendo commits {done}/{total}…")

        stats = await index_history(store, repo, rev, since, options, on_batch=progress)
    err.print(
        f"{stats.walked} commits recorridos ({stats.commits - stats.walked} ya estaban) · "
        f"{stats.blobs} blobs leídos · {stats.sightings} ubicaciones nuevas"
    )
