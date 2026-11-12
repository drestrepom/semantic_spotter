# semantic-spotter

Detecta código duplicado en Go y TypeScript dentro de un repositorio git, y avisa cuando un cambio introduce una copia de algo que ya existía.

## Unidades de código

**Snippet**:
Una función o un bloque extraído de un archivo, identificado por su ubicación (ruta y rango) en un blob de git y por el hash de su contenido.
_Avoid_: fragmento, chunk

**Ocurrencia**:
Un snippet en una ruta concreta de un árbol; el mismo contenido en dos rutas son dos ocurrencias del mismo hash.
_Avoid_: instancia, copia (cuando no se habla de un clon)

**Ocurrencia cambiada**:
Una ocurrencia cuyo par (ruta, hash) aparece más veces en head que en la base; excluye los movimientos y los corrimientos de línea.
_Avoid_: snippet nuevo, diff

## Clones

**Clon**:
Dos o más ocurrencias que se parecen lo suficiente para considerarse duplicadas, clasificadas por tipo.
_Avoid_: duplicado semántico (cuando no es T4)

**T1**:
Clon con contenido idéntico salvo espacios y comentarios.

**T2**:
Clon idéntico salvo nombres de identificadores y literales.

**T3**:
Clon con estructura casi igual (alta similitud Jaccard de tokens normalizados), con sentencias agregadas o quitadas.

**T4**:
Clon que hace lo mismo con código distinto, detectado solo por similitud de embeddings.
_Avoid_: clon semántico (para T1–T3)

**Grupo**:
Conjunto de ocurrencias unidas por clones entre sí, que se reporta como una unidad.

**Marca de ignorar**:
Comentario `semantic-spotter:ignore -- <motivo>` inmediatamente antes de una función o bloque, que declara esa copia como intencional; cubre los bloques internos y no se hereda a nuevas copias.
_Avoid_: nolint, supresión, excepción

## Historia

**Avistamiento**:
El registro de que una ocurrencia estaba presente en un commit, aunque no se haya creado en él.
_Avoid_: aparición

**Primer avistamiento**:
El commit más antiguo de la historia recorrida en el que aparece una ocurrencia (ruta y hash).
_Avoid_: autor, origen
