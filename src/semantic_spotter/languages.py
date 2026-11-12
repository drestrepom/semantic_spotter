from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

import tree_sitter_go
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

# Devuelve el nombre si el nodo es una función con nombre; None en otro caso.
FunctionNamer = Callable[[Node], str | None]


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    language: Language
    function_name: FunctionNamer
    # tipo de nodo -> etiqueta corta del bloque
    block_kinds: dict[str, str]
    # tipos de nodo padre bajo los cuales un `if` es un `else if` de la cadena
    else_if_parents: frozenset[str]
    # hojas que la normalización estructural reemplaza por `ID`
    identifier_kinds: frozenset[str]
    # nodos (con su subárbol) que la normalización reemplaza por `LIT`
    literal_kinds: frozenset[str]

    def parser(self) -> Parser:
        return Parser(self.language)


def _text(node: Node | None) -> str | None:
    if node is None or node.text is None:
        return None
    return node.text.decode("utf-8", "replace")


# --- Go ---------------------------------------------------------------------


def _go_receiver_type(method: Node) -> str:
    receiver = method.child_by_field_name("receiver")
    if receiver is not None:
        for param in receiver.named_children:
            if param.type == "parameter_declaration":
                type_text = _text(param.child_by_field_name("type")) or ""
                return type_text.lstrip("*").split("[", 1)[0]
    return "?"


def _go_function_name(node: Node) -> str | None:
    if node.type == "function_declaration":
        return _text(node.child_by_field_name("name"))
    if node.type == "method_declaration":
        name = _text(node.child_by_field_name("name"))
        return f"{_go_receiver_type(node)}.{name}" if name else None
    return None


GO = LanguageSpec(
    name="go",
    language=Language(tree_sitter_go.language()),
    function_name=_go_function_name,
    block_kinds={
        "for_statement": "for",
        "if_statement": "if",
        "expression_switch_statement": "switch",
        "type_switch_statement": "switch",
        "select_statement": "select",
    },
    else_if_parents=frozenset({"if_statement"}),
    identifier_kinds=frozenset(
        {"identifier", "field_identifier", "type_identifier", "package_identifier", "label_name"}
    ),
    literal_kinds=frozenset(
        {
            "interpreted_string_literal",
            "raw_string_literal",
            "int_literal",
            "float_literal",
            "imaginary_literal",
            "rune_literal",
            "true",
            "false",
            "nil",
            "iota",
        }
    ),
)


# --- TypeScript -------------------------------------------------------------

_TS_CLASS_TYPES = frozenset({"class_declaration", "abstract_class_declaration", "class"})
_TS_FUNCTION_VALUES = frozenset({"arrow_function", "function_expression", "generator_function"})


def _ts_class_name(node: Node) -> str:
    parent = node.parent
    while parent is not None:
        if parent.type in _TS_CLASS_TYPES:
            return _text(parent.child_by_field_name("name")) or "<anon>"
        parent = parent.parent
    return "<anon>"


def _ts_function_name(node: Node) -> str | None:
    if node.type in ("function_declaration", "generator_function_declaration"):
        return _text(node.child_by_field_name("name"))
    if node.type == "method_definition":
        name = _text(node.child_by_field_name("name"))
        return f"{_ts_class_name(node)}.{name}" if name else None
    if node.type in _TS_FUNCTION_VALUES:
        # Solo cuenta si está asignada a algo con nombre: `const f = () => {}`
        # o un campo de clase `f = () => {}`. Los callbacks anónimos se ignoran.
        parent = node.parent
        if parent is None or parent.child_by_field_name("value") != node:
            return None
        name = _text(parent.child_by_field_name("name"))
        if name is None:
            return None
        if parent.type == "variable_declarator":
            return name
        if parent.type == "public_field_definition":
            return f"{_ts_class_name(parent)}.{name}"
    return None


_TS_BLOCK_KINDS = {
    "for_statement": "for",
    "for_in_statement": "for",
    "while_statement": "while",
    "do_statement": "do",
    "if_statement": "if",
    "switch_statement": "switch",
    "try_statement": "try",
}

_TS_IDENTIFIER_KINDS = frozenset(
    {
        "identifier",
        "type_identifier",
        "property_identifier",
        "private_property_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
        "statement_identifier",
    }
)
_TS_LITERAL_KINDS = frozenset(
    {"string", "template_string", "number", "regex", "true", "false", "null", "undefined"}
)

TYPESCRIPT = LanguageSpec(
    name="typescript",
    language=Language(tree_sitter_typescript.language_typescript()),
    function_name=_ts_function_name,
    block_kinds=_TS_BLOCK_KINDS,
    else_if_parents=frozenset({"else_clause"}),
    identifier_kinds=_TS_IDENTIFIER_KINDS,
    literal_kinds=_TS_LITERAL_KINDS,
)

TSX = LanguageSpec(
    name="tsx",
    language=Language(tree_sitter_typescript.language_tsx()),
    function_name=_ts_function_name,
    block_kinds=_TS_BLOCK_KINDS,
    else_if_parents=frozenset({"else_clause"}),
    identifier_kinds=_TS_IDENTIFIER_KINDS,
    literal_kinds=_TS_LITERAL_KINDS,
)

_BY_EXTENSION = {
    ".go": GO,
    ".ts": TYPESCRIPT,
    ".mts": TYPESCRIPT,
    ".cts": TYPESCRIPT,
    ".tsx": TSX,
}


def spec_for(path: str) -> LanguageSpec | None:
    return _BY_EXTENSION.get(PurePosixPath(path).suffix)
