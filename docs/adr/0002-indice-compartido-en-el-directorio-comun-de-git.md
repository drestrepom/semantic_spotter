# El índice vive en el directorio común de git, compartido entre worktrees, con lock

El índice se guarda en `$(git rev-parse --git-common-dir)/semantic-spotter/` y no en la raíz del worktree, porque en inity hay decenas de worktrees (la mayoría de agentes de Claude) que comparten casi todos sus blobs: por worktree, cada uno pagaba una indexación en frío (~9 s, ~16 MB) para re-extraer blobs que otro ya había extraído, justo en el primer commit, donde corre el hook. Compartirlo significa que varios procesos escriben a la vez, así que las escrituras se serializan con un `flock` exclusivo sobre un archivo del índice y, tras tomarlo, se vuelve a consultar qué falta; las escrituras son idempotentes porque el mismo blob con la misma clave produce siempre las mismas filas. El lock se toma con un único `async with store.writing()` explícito en cada punto de escritura, sin reentrada; leer y abrir un índice ya creado no lo toman, así que un `check` en caliente nunca espera a la extracción en frío de otro worktree. Quien espera el lock más de 1 s lo avisa por stderr y a los 60 s falla con exit 2, en vez de colgar el hook. `SEMANTIC_SPOTTER_DB` reemplaza la ruta para pruebas y, más adelante, para un índice remoto en CI; una ruta relativa se resuelve contra el repo.

## Considered Options

- **Por worktree** (lo anterior): sin concurrencia, pero con el costo en frío repetido en cada worktree y sin compartir embeddings.
- **Configurable con el worktree como default**: deja el costo en el caso común.
