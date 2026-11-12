# El índice vive en el directorio común de git, compartido entre worktrees, con lock

El índice se guarda en `$(git rev-parse --git-common-dir)/semantic-spotter/` y no en la raíz del worktree, porque en inity hay decenas de worktrees (la mayoría de agentes de Claude) que comparten casi todos sus blobs: por worktree, cada uno pagaba una indexación en frío (~9 s, ~16 MB) para re-extraer blobs que otro ya había extraído, justo en el primer commit, donde corre el hook. Compartirlo significa que varios procesos escriben a la vez, así que las escrituras se serializan con un `flock` exclusivo sobre un archivo del índice y, tras tomarlo, se vuelve a consultar qué falta; las escrituras son idempotentes porque el mismo blob con la misma clave produce siempre las mismas filas. `SEMANTIC_SPOTTER_DB` reemplaza la ruta para pruebas y, más adelante, para un índice remoto en CI.

## Considered Options

- **Por worktree** (lo anterior): sin concurrencia, pero con el costo en frío repetido en cada worktree y sin compartir embeddings.
- **Configurable con el worktree como default**: deja el costo en el caso común.
