---
labels: [ready-for-agent]
---

# semantic-spotter como flake de Nix y hook de duplicados en inity

## Problem Statement

En inity se escribe código duplicado —funciones copiadas entre módulos, plantillas repetidas— y nadie se entera hasta que alguien corre `semantic-spotter` a mano. La herramienta existe y funciona, pero vive como un paquete de Python suelto: inity solo instala herramientas externas como flakes de Nix, así que no hay forma reproducible de usarla allá. Además, buena parte del código de inity lo escriben agentes de Claude en worktrees propios, y un agente no tiene cómo saber que lo que acaba de escribir ya existía.

Hoy, además:

- El índice vive en la raíz de cada worktree, así que cada uno de los ~36 worktrees de inity (la mayoría de agentes) paga una indexación en frío (~9 s, ~16 MB) y re-extrae blobs idénticos a los que ya extrajeron los demás.
- No hay forma de declarar que una copia es intencional: la única salida ante un falso positivo sería saltarse el hook entero.
- Correrlo en CI costaría tokens de Voyage en cada corrida porque no hay caché compartida de embeddings.

## Solution

`semantic-spotter` se publica como un flake de Nix construido con uv2nix desde su propio repositorio público. inity lo consume como input del flake y lo corre en un hook de pre-commit de lefthook que revisa solo lo stageado, en modo estructural (T1–T3, sin embeddings, sin tokens, sin API key):

- **T1 y T2 bloquean el commit**: son copias literales o con nombres cambiados, casi nunca falsos positivos.
- **T3 solo avisa**: detecta plantillas reales pero también estructuras parecidas por convención.
- El mensaje de error dice qué hacer: extraer la lógica compartida o, si la copia es intencional, poner una **marca de ignorar** con su motivo. Un agente que hace commit recibe ese mensaje, y con él corrige.
- El índice pasa al directorio común de git, compartido por todos los worktrees, de modo que un worktree nuevo encuentra casi todo ya extraído.
- No hay CI todavía: entra cuando exista una caché compartida de embeddings.

## User Stories

### Dev o agente que hace commit en inity

1. Como agente de Claude que hace commit en inity, quiero que el commit falle si mi cambio introduce un T1 de una función existente, para enterarme de que la copié y reutilizar la original.
2. Como agente de Claude, quiero que el commit falle si mi cambio introduce un T2 (misma estructura con otros nombres o literales), para no duplicar lógica renombrando variables.
3. Como agente de Claude, quiero que el mensaje de fallo nombre las ubicaciones de cada miembro del grupo, para ir directo al código original.
4. Como agente de Claude, quiero que el mensaje de fallo diga explícitamente las dos salidas —extraer lo compartido o marcar la copia con `semantic-spotter:ignore -- <motivo>`—, para corregir sin tener que leer documentación.
5. Como agente de Claude, quiero que un T3 se imprima como aviso sin bloquear el commit, para considerarlo sin que una similitud por convención me frene.
6. Como dev de inity, quiero que el hook solo corra cuando el commit toca archivos Go o TypeScript, para no pagar nada en commits de docs, Nix o SQL.
7. Como dev de inity, quiero que el hook no necesite `VOYAGE_API_KEY` ni gaste tokens, para que funcione en cualquier máquina y worktree sin configuración.
8. Como dev de inity, quiero que el hook analice solo las ocurrencias cambiadas por lo stageado, para que una función que solo se movió de línea o de archivo no dispare nada.
9. Como dev de inity, quiero que el hook compare lo cambiado contra todo el snapshot stageado, para detectar también copias entre dos archivos del mismo commit.
10. Como dev de inity, quiero que el hook tarde menos de un segundo en un worktree cuyo índice ya está caliente, para que no estorbe el flujo de commits.
11. Como dev de inity, quiero poder saltarme el hook como cualquier otro de lefthook (`LEFTHOOK_EXCLUDE` o `--no-verify`), para no quedar bloqueado ante una falla de la herramienta.
12. Como dev de inity, quiero que si la herramienta falla por un error (no por duplicados) salga con un código distinto y un mensaje claro, para no confundir un bug con un duplicado.

### Marca de ignorar

13. Como agente o dev, quiero marcar una función con `// semantic-spotter:ignore -- <motivo>` en la línea anterior, para declarar que esa copia es intencional y que el hook la deje pasar.
14. Como agente o dev, quiero que la marca de una función cubra sus bloques internos, para no tener que marcar cada `for` o `if` que también se reporta como clon.
15. Como revisor de PR, quiero que una marca sin motivo después de `--` no tenga efecto, para que nadie silencie duplicados sin justificar por qué.
16. Como revisor de PR, quiero que copiar una función marcada a otro lugar sin marcar la nueva copia falle, para que la marca no se convierta en permiso para seguir copiando.
17. Como revisor de PR, quiero que un grupo solo se omita cuando todos sus miembros cambiados están marcados, para que un miembro nuevo sin marca siga fallando aunque el original esté marcado.
18. Como revisor de PR, quiero que los miembros no cambiados del grupo no cuenten para la omisión, para que el agente no tenga que marcar código ajeno que no tocó.
19. Como revisor de PR, quiero poder auditar todas las marcas con un `grep` de `semantic-spotter:ignore`, para revisar sus motivos igual que los `nolint`.
20. Como dev, quiero que agregar una marca no cambie el hash del snippet marcado, para que marcar no se vea como un cambio de contenido ni dispare otra detección.
21. Como dev, quiero que la marca también silencie el aviso de T3 del grupo, para que una plantilla intencional documentada no siga haciendo ruido.

### Exclusiones de inity

22. Como dev de inity, quiero que las migraciones (`apps/*/migrations/`) queden fuera del análisis, para que su parecido por diseño no bloquee commits.
23. Como dev de inity, quiero que los componentes de shadcn (`**/components/ui/`) queden fuera, para que el código copiado del generador no cuente como duplicado.
24. Como revisor de PR, quiero que esas exclusiones vivan en un `.semantic-spotterignore` commiteado en inity con un comentario por patrón, para que agregar una exclusión pase por revisión y el hook, `scan` y el CI futuro usen la misma lista.
25. Como dev, quiero que los tests y el código generado sigan fuera por defecto sin configuración, para no repetir reglas que la herramienta ya tiene.

### Índice compartido

26. Como agente en un worktree nuevo de inity, quiero que el índice viva en el directorio común de git, para encontrar ya extraídos los blobs que extrajeron otros worktrees.
27. Como dev de inity, quiero que no se cree un `.semantic-spotter/` en la raíz de cada worktree, para no tener ~36 copias de ~16 MB.
28. Como agente que hace commit al mismo tiempo que otro agente, quiero que las escrituras al índice compartido estén protegidas por un lock, para que dos `check` simultáneos no corrompan ni pierdan datos.
29. Como agente, quiero que dos `check` concurrentes sobre el mismo índice terminen los dos con el mismo resultado que si corrieran en serie, para que el resultado del hook no dependa de quién escribió primero.
30. Como mantenedor, quiero poder apuntar el índice a otra ruta con `SEMANTIC_SPOTTER_DB`, para usar un índice distinto en pruebas y, más adelante, uno remoto en CI.
31. Como dev de inity, quiero que el `.semantic-spotter/` viejo de la raíz se pueda borrar sin perder nada que importe, para no dejar basura tras la migración.

### Flake de Nix

32. Como mantenedor de inity, quiero agregar semantic-spotter como input del flake con `follows` para `nixpkgs`, `pyproject-nix`, `uv2nix` y `pyproject-build-systems`, para no duplicar esas dependencias en el lock.
33. Como mantenedor de inity, quiero que la versión de semantic-spotter quede fijada solo por el `flake.lock` de inity, para actualizarla con `nix flake update semantic-spotter` como cualquier otro input.
34. Como dev de inity, quiero correr la herramienta con `nix run .#semantic-spotter`, para usar exactamente la versión del flake y no un binario suelto que diverja.
35. Como dev de inity en x86_64-linux, aarch64-linux o aarch64-darwin, quiero que el paquete construya en mi sistema, para que el hook funcione en mi máquina y en CodeBuild Graviton.
36. Como mantenedor de semantic-spotter, quiero que el paquete exponga solo el binario `semantic-spotter`, para no filtrar el entorno de Python a quien lo instale.
37. Como mantenedor, quiero que el paquete se construya desde ruedas precompiladas sobre Python 3.14, para no compilar tree-sitter, lancedb ni pyarrow desde fuente.
38. Como mantenedor, quiero que `nix flake check` corra la suite de pytest con git disponible, para que el empaquetado no rompa el comportamiento.
39. Como mantenedor, quiero un devShell del flake con Python, uv y git, para trabajar en la herramienta sin instalar nada fuera de Nix.
40. Como mantenedor, quiero que `nix run github:drestrepom/semantic_spotter -- --help` funcione sin clonar el repo, para que cualquiera la pruebe.

### Documentación

41. Como lector futuro de semantic-spotter, quiero un ADR que explique por qué el índice es por blob y no guarda texto, para no "arreglarlo" agregando una tabla de snippets por commit.
42. Como lector futuro, quiero un ADR que explique por qué el índice vive en el directorio común de git y usa un lock, para entender por qué no está en el worktree.
43. Como dev o agente de inity frenado por el hook, quiero un ADR en inity que explique por qué T1 y T2 bloquean, por qué T3 avisa, cómo funciona la marca y por qué no hay CI todavía, para entender la regla antes de discutirla.
44. Como lector, quiero un glosario con los términos del dominio (snippet, ocurrencia, ocurrencia cambiada, T1–T4, grupo, marca de ignorar, avistamiento), para hablar de lo mismo en issues, ADRs y código.

## Implementation Decisions

### Marca de ignorar

- Se detecta **al extraer**: es el comentario inmediatamente anterior al nodo de la función o bloque (hermano previo en el árbol de tree-sitter, sin líneas de código entre medio). Funciona igual en Go y TypeScript (`//`).
- Formato: `semantic-spotter:ignore -- <motivo>`. Sin motivo no vacío después de `--`, la marca es inválida y se trata como si no existiera.
- Se guarda como una nueva columna `ignore_reason` (string, vacío si no hay marca válida) en la tabla de ubicaciones por blob. No se guarda ningún otro texto.
- Los bloques heredan la marca de la función que los contiene (o del bloque que los contiene): se resuelve al extraer, de modo que cada ubicación ya trae su `ignore_reason` efectivo.
- El comentario queda fuera del rango de bytes del nodo, así que no altera `content_hash`, `norm_hash` ni shingles.
- Como cambia lo que produce la extracción, `EXTRACT_VERSION` sube a 4: la siguiente corrida re-extrae todo una vez. Los embeddings se reutilizan porque están indexados por `content_hash`, que no cambia.
- `Snippet` gana un campo con el motivo de la marca (vacío por defecto).
- En `check`, después de agrupar: un grupo se descarta si **todos sus miembros cambiados** tienen marca válida. Los miembros no cambiados no cuentan. Esto aplica tanto a los grupos que fallan como a los que solo avisan. En `scan`, los grupos cuyos miembros están todos marcados se omiten del reporte.
- Los grupos descartados por marca no cuentan para el exit code.

### Mensaje de fallo

- Cuando `check` sale con 1, imprime al final (stderr) una instrucción accionable: extraer la lógica compartida y reutilizarla, o, si la copia es intencional, agregar `// semantic-spotter:ignore -- <motivo>` en la línea anterior a la función o bloque nuevo.
- El reporte por grupo sigue mostrando ubicación, tipo de clon, marca de "nuevo" y primer avistamiento como hoy.

### Índice compartido

- La ruta por defecto del índice pasa a ser `<directorio común de git>/semantic-spotter/` (resultado de `git rev-parse --git-common-dir`, resuelto respecto del repo). Es el mismo para todos los worktrees.
- `SEMANTIC_SPOTTER_DB`, si está definida, la reemplaza.
- Ya no se crea `.semantic-spotter/` en la raíz del worktree. El índice lleva siempre un `.gitignore` con `*`, por si `SEMANTIC_SPOTTER_DB` lo pone dentro de un worktree; una ruta relativa en esa variable se resuelve contra el repo.
- Las escrituras se serializan con un lock exclusivo de archivo (`flock`) sobre un archivo dentro del directorio del índice. Las lecturas no toman el lock; la consistencia de lectura ya es fuerte.
- El lock cubre cada operación de escritura completa (registrar un árbol, guardar blobs extraídos, guardar historia, guardar embeddings). Antes de extraer, tras tomar el lock, se vuelve a consultar qué blobs faltan, para no re-extraer lo que otro proceso acaba de guardar.
- Las escrituras son idempotentes: el mismo blob con la misma clave de extracción siempre produce las mismas filas, y el patrón borrar-y-agregar por blob se mantiene.
- No se migra el índice viejo: el de la raíz de inity simplemente se borra y el nuevo se construye en la primera corrida.

### Flake de Nix

- Vive en la raíz del repo de semantic-spotter, que será público como `github:drestrepom/semantic_spotter`.
- Inputs: `nixpkgs`, `pyproject-nix`, `uv2nix`, `pyproject-build-systems`, declarados de forma que el consumidor pueda hacer `follows` de los cuatro.
- Construcción con uv2nix desde `uv.lock`, intérprete `python314`, `sourcePreference = "wheel"`. El backend de build del proyecto es `uv_build`, que viene de `pyproject-build-systems`.
- Sistemas: x86_64-linux, aarch64-linux, aarch64-darwin.
- Salidas por sistema:
  - `packages.default`: aplicación que expone solo el binario `semantic-spotter` (no el entorno virtual completo).
  - `apps.default`: apunta a ese binario.
  - `checks.default`: corre pytest contra el entorno con dependencias de desarrollo, con git en el PATH.
  - `devShells.default`: Python, uv y git, con el proyecto en modo editable.
- Sin tags ni releases: la versión la fija el lock del consumidor.

### Integración en inity

- Input `semantic-spotter` en el flake de inity con `follows` de los cuatro inputs hacia los que inity ya tiene.
- Se expone como `.#semantic-spotter` siguiendo la convención de inity para herramientas (`nix/pkgs/<nombre>`), que reexporta el paquete del input.
- Comando `duplicados` en el pre-commit de `lefthook.yml`, con `glob` de archivos Go y TypeScript:
  `nix run .#semantic-spotter -- check --staged --structural --fail-on t1,t2 --t3-jaccard 0.8`
  con un comentario que explique la regla y remita al ADR.
- `.semantic-spotterignore` en la raíz de inity con `apps/*/migrations/` y `**/components/ui/`, cada uno con su porqué.
- ADR 0294 en inity: duplicados en pre-commit y no en CI hasta que exista caché compartida de embeddings; T1/T2 bloquean, T3 avisa; marca de ignorar.
- Se borra el `.semantic-spotter/` de la raíz de inity.
- Sin cambios en los workflows de CI de inity.

### Documentación en semantic-spotter

- ADR 0001: índice por blob sin texto.
- ADR 0002: índice compartido en el directorio común de git con lock.
- El glosario del repo ya define los términos usados aquí.

## Testing Decisions

- Un buen test aquí ejercita el comportamiento externo: se arma un repo git temporal, se corre el CLI y se miran el exit code, la salida y los efectos observables (dónde quedó el índice, qué se re-extrajo). No se testean helpers internos del lock ni de la detección de comentarios por separado.
- **Seam principal: el CLI `check` sobre un repo git temporal**, con el fixture de repo de los tests de CLI existentes y el embedder falso. Casos:
  - copia marcada con motivo → exit 0;
  - marca sin motivo → exit 1;
  - copia sin marca de una función marcada → exit 1;
  - la marca de la función cubre sus bloques internos → exit 0;
  - grupo con un miembro cambiado marcado y otro no → exit 1;
  - agregar solo la marca a una función existente no genera ocurrencias cambiadas;
  - un T3 marcado no aparece como aviso;
  - el stderr de un fallo contiene las dos salidas (extraer o marcar);
  - con `git worktree add`, el índice queda en el directorio común de git, el worktree no tiene `.semantic-spotter/`, y el `check` en el segundo worktree no re-extrae blobs ya extraídos (con el spy de extracción que ya usan los tests de pipeline);
  - con `SEMANTIC_SPOTTER_DB`, el índice va a esa ruta;
  - dos `check` en procesos paralelos sobre el mismo índice terminan ambos con el exit code y grupos esperados.
- **Seam de empaquetado: el flake.** `nix flake check` (que corre la suite) y `nix run . -- --help` en verde. Es humo del empaquetado, no de comportamiento.
- **Verificación manual en inity** (sin tests nuevos en inity): en un worktree temporal, stagear una copia de una función existente y correr `lefthook run pre-commit` → falla con el mensaje; marcarla con motivo → pasa.
- Prior art: los tests de CLI de `check` (copia renombrada, movimientos y corrimientos, `--staged`, `--fail-on`, sin API key), el test de "la DB no guarda texto" y el spy de conteo de extracciones de los tests de pipeline.

## Out of Scope

- Correr el chequeo de duplicados en el CI de inity, y un presupuesto de duplicación por módulo al estilo nolint-budget.
- Caché compartida (S3 u otra) de índice y embeddings; T4 en el hook.
- Un modo `scan --structural`.
- Verificación de grupos con un LLM, set de calibración y salida SARIF.
- Contar o limitar las marcas de ignorar.
- Releases versionadas o tags del flake.
- Migrar el contenido del índice viejo al nuevo.

## Further Notes

- Mediciones de referencia en inity: indexación en frío del árbol de HEAD ~9 s y ~16 MB; pares estructurales sobre ~15.6k snippets en ~4 s; historia completa (5.8k commits) en ~43 s y ~5 MB, incremental ~1 s.
- En los últimos 77 commits de inity, `check --structural --t3-jaccard 0.8` habría encontrado 1 grupo T1, 7 T2 y 19 T3: el hook habría bloqueado 8 veces y avisado 19.
- T4 sin caché costaría ~5.7M tokens por corrida completa; con caché, ~135k tokens cada 77 commits. Por eso el CI espera a la caché.
- El orden sugerido es: semantic-spotter (marca, índice compartido, mensaje, flake, ADRs) → primer commit y repo público en GitHub → integración en inity.
