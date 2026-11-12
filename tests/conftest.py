import asyncio
import hashlib
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest


class FakeEmbedder:
    """Embeddings deterministas por feature hashing de tokens.

    Dos textos con los mismos identificadores producen vectores parecidos, lo
    que basta para probar el pipeline sin red.
    """

    model = "fake"
    dim = 64

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        return [len(re.findall(r"\w+", t)) for t in texts]

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        self.embedded.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return out


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def make_repo(tmp_path: Path):
    """Crea un repo git con los archivos dados y devuelve (ruta, sha)."""

    def _make(files: dict[str, str]) -> tuple[Path, str]:
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        if not (repo / ".git").exists():
            git(repo, "init", "-q")
            git(repo, "config", "user.email", "t@example.com")
            git(repo, "config", "user.name", "t")
        for rel, content in files.items():
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "c")
        return repo, git(repo, "rev-parse", "HEAD")

    return _make
