# El índice es por blob de git y no guarda el texto de los snippets

El índice no tiene una fila por snippet y commit: guarda, por cada blob de git, la ubicación (rango de bytes y líneas) y el hash de cada snippet, y un commit se resuelve a sus snippets uniendo `commits → tree_files → blob_snippets → contents`. Como los commits comparten casi todos sus blobs, indexar un commit nuevo solo extrae los blobs que cambiaron, y guardar historia completa cuesta megabytes en vez de una copia del repo por commit. El texto, cuando hace falta (solo para embeber hashes que todavía no tienen vector), se lee de git con `git cat-file --batch` y se corta por bytes: git ya lo guarda, y duplicarlo en el índice lo haría crecer con cada commit.

## Considered Options

- **Una fila por (commit, snippet) con el texto** (el esquema v2): simple de consultar, pero ~15 mil filas con texto por commit en inity aunque cambie un solo archivo, y sin forma barata de saber qué cambió.
- **Por archivo en vez de por blob**: no aprovecha que un mismo blob aparece en muchas rutas y commits.

## Consequences

- El índice no se puede leer sin el repo git al lado: mostrar o embeber un snippet requiere sus blobs.
- Las opciones de extracción que cambian los resultados forman parte de la clave del blob (`extract_key`), y un cambio en el extractor sube `EXTRACT_VERSION` y re-extrae todo una vez. Los embeddings sobreviven porque se indexan por `content_hash`.
