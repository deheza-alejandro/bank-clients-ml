# Configuración

La configuración central está en `src/bank_clients_ml/config.py`. La clase
`Settings` agrupa las rutas, los nombres de las columnas y los parámetros
globales, como `random_state` y `n_jobs`. Los valores por defecto están
definidos en ese archivo y se pueden redefinir con variables de entorno.

Las variables de la aplicación usan el prefijo `ML_`. Los campos anidados se
separan con `__`. Algunos ejemplos de variables que se pueden definir en un
`.env`:

```bash
ML_DEBUG=true
ML_RANDOM_STATE=42
ML_N_JOBS=4
ML_PATHS__RAW_DATA_FILE_NAME=example_data.parquet
ML_DATASET__TARGET=Target
```

El listado completo de variables y sus valores por defecto está en `Settings`,
`DirectoryPaths`, `DatasetColumns` y `AnalysisColumns`, en
`src/bank_clients_ml/config.py`. Las rutas `raw_data` e `images` se calculan a
partir de `paths.project`; `raw_data` es
`paths.project / "data" / paths.raw_data_file_name`. No se configuran de forma
independiente. En el uso normal no hace falta definir `ML_PATHS__PROJECT`: se
calcula desde la ubicación de `config.py` y en el contenedor de desarrollo
corresponde a `/app`.

También se puede crear un archivo `.env` en la raíz del proyecto. Es opcional
y está ignorado por Git. Del archivo solo se leen las variables con prefijo
`ML_`, así que las variables de Docker Compose pueden convivir en el mismo
`.env`. En el contenedor de desarrollo se lee desde `/app/.env` mediante el
montaje del proyecto; no se copia dentro de la imagen.

Las variables de entorno del sistema tienen prioridad sobre el `.env`, y este
sobre los valores por defecto.

Si se reemplaza el conjunto de datos, conservar el nombre
`example_data.parquet` o definir `ML_PATHS__RAW_DATA_FILE_NAME`. Después de
modificar el `.env` o las variables de entorno, reiniciar el proceso o el
contenedor. `Settings.settings_cached()` guarda la configuración en caché y un
proceso ya iniciado no detecta los cambios. Para invalidar esa caché, llamar a
`Settings.settings_clear_cache()`.
