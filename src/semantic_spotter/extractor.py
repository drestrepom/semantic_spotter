import hashlib
import textwrap

from tree_sitter import Node

from semantic_spotter.languages import LanguageSpec
from semantic_spotter.models import Snippet, Unit

SHINGLE_SIZE = 5


def snippet_text(source: bytes, start_byte: int, end_byte: int) -> str:
    """Texto de `source[start_byte:end_byte]` sin la indentación común.

    Si el rango empieza después de solo espacios en su línea, esos espacios se
    incluyen para que `dedent` pueda alinear la primera línea con el resto.
    """
    line_start = source.rfind(b"\n", 0, start_byte) + 1
    prefix = source[line_start:start_byte]
    raw = source[start_byte:end_byte]
    if not prefix.strip():
        raw = prefix + raw
    return textwrap.dedent(raw.decode("utf-8", "replace"))


def _non_empty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_tokens(node: Node, spec: LanguageSpec) -> list[str]:
    """Hojas del nodo con identificadores → `ID` y literales → `LIT`, sin comentarios.

    Dos unidades con la misma secuencia difieren solo en nombres o valores (clon T2).
    """
    tokens: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "comment":
            continue
        if current.type in spec.literal_kinds:
            tokens.append("LIT")
        elif current.type in spec.identifier_kinds:
            tokens.append("ID")
        elif current.child_count == 0:
            tokens.append(current.type)
        else:
            stack.extend(reversed(current.children))
    return tokens


def shingles(tokens: list[str], k: int = SHINGLE_SIZE) -> tuple[int, ...]:
    """Hashes int64 de los k-gramas de tokens (un único k-grama si hay menos de k)."""
    grams = {" ".join(tokens[i : i + k]) for i in range(max(1, len(tokens) - k + 1))}
    return tuple(
        sorted(
            int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), signed=True)
            for g in grams
        )
    )


def extract(
    spec: LanguageSpec,
    file_path: str,
    source: bytes,
    blob_oid: str,
    *,
    min_func_lines: int = 3,
    min_block_lines: int = 6,
    fingerprint: bool = True,
) -> list[Snippet]:
    """Funciones y bloques del archivo; sin `fingerprint` no calcula `norm_hash` ni shingles."""
    tree = spec.parser().parse(source)
    snippets: list[Snippet] = []

    def emit(node: Node, unit: Unit, kind: str, name: str, min_lines: int) -> None:
        text = snippet_text(source, node.start_byte, node.end_byte)
        if _non_empty_lines(text) < min_lines:
            return
        tokens = normalized_tokens(node, spec) if fingerprint else []
        snippets.append(
            Snippet(
                content_hash=content_hash(text),
                blob_oid=blob_oid,
                file_path=file_path,
                language=spec.name,
                unit=unit,
                kind=kind,
                name=name,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                norm_hash=content_hash(" ".join(tokens)) if fingerprint else "",
                shingles=shingles(tokens) if fingerprint else (),
            )
        )

    # Recorrido iterativo en orden de fuente; `enclosing` es la función con
    # nombre más cercana, usada para nombrar los bloques.
    stack: list[tuple[Node, str | None]] = [(tree.root_node, None)]
    while stack:
        node, enclosing = stack.pop()
        func_name = spec.function_name(node)
        if func_name is not None:
            kind = "method" if "." in func_name else "func"
            emit(node, "function", kind, func_name, min_func_lines)
            enclosing = func_name
        elif node.type in spec.block_kinds and not (
            node.type == "if_statement"
            and node.parent is not None
            and node.parent.type in spec.else_if_parents
        ):
            kind = spec.block_kinds[node.type]
            line = node.start_point.row + 1
            name = f"{enclosing or '<top>'} › {kind}@L{line}"
            emit(node, "block", kind, name, min_block_lines)
        stack.extend((child, enclosing) for child in reversed(node.named_children))

    return snippets
