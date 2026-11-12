from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np
import voyageai

# Límites de la API de Voyage por request (120K es el tope más conservador publicado).
MAX_BATCH_TEXTS = 1000
MAX_BATCH_TOKENS = 120_000
MAX_CONTEXT_TOKENS = 32_000


class Embedder(Protocol):
    model: str
    dim: int

    def token_counts(self, texts: Sequence[str]) -> list[int]: ...

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Matriz float32 de forma (len(texts), dim)."""
        ...


def batches(
    token_counts: Sequence[int],
    *,
    max_texts: int = MAX_BATCH_TEXTS,
    max_tokens: int = MAX_BATCH_TOKENS,
) -> Iterator[range]:
    """Rangos contiguos de índices que respetan ambos límites por request."""
    start, tokens = 0, 0
    for i, count in enumerate(token_counts):
        if i > start and (i - start >= max_texts or tokens + count > max_tokens):
            yield range(start, i)
            start, tokens = i, 0
        tokens += count
    if start < len(token_counts):
        yield range(start, len(token_counts))


class VoyageEmbedder:
    def __init__(
        self,
        model: str = "voyage-code-4",
        dim: int = 1024,
        *,
        api_key: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.dim = dim
        # El cliente reintenta con backoff exponencial ante rate limits.
        self._client = voyageai.AsyncClient(api_key=api_key, max_retries=max_retries)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        # El tokenizer es local (HF tokenizers, en Rust): no hay I/O que esperar.
        encodings = self._client.tokenize(list(texts), self.model)
        # La API trunca lo que excede el contexto, así que no se cobra más.
        return [min(len(e.ids), MAX_CONTEXT_TOKENS) for e in encodings]

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        result = await self._client.embed(
            list(texts),
            model=self.model,
            input_type="document",
            output_dimension=self.dim,
        )
        return np.asarray(result.embeddings, dtype=np.float32)
