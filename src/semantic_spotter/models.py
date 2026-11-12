from dataclasses import dataclass
from typing import Literal

Unit = Literal["function", "block"]


@dataclass(frozen=True, slots=True)
class TreeFile:
    """Un archivo de un árbol de git, con el lenguaje con el que se extrae."""

    path: str
    blob_oid: str
    language: str


@dataclass(frozen=True, slots=True)
class Snippet:
    """Una unidad extraída (función o bloque) en una ruta de un árbol de git.

    No guarda el texto: se recupera del blob con `extractor.snippet_text`.
    """

    content_hash: str
    blob_oid: str
    file_path: str
    language: str
    unit: Unit
    kind: str
    name: str
    start_line: int  # 1-based, inclusiva
    end_line: int  # 1-based, inclusiva
    start_byte: int
    end_byte: int
    # sha256 de los tokens con identificadores y literales normalizados (T2)
    norm_hash: str = ""
    # hashes de k-gramas de tokens normalizados, para Jaccard (T3)
    shingles: tuple[int, ...] = ()
    # motivo de la marca de ignorar efectiva (propia o del contenedor); "" sin marca
    ignore_reason: str = ""

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def lines(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def size(self) -> int:
        return self.end_byte - self.start_byte

    def contains(self, other: "Snippet") -> bool:
        return (
            self.file_path == other.file_path
            and self.start_byte <= other.start_byte
            and other.end_byte <= self.end_byte
        )

    def overlaps(self, other: "Snippet") -> bool:
        return (
            self.file_path == other.file_path
            and self.start_byte < other.end_byte
            and other.start_byte < self.end_byte
        )
