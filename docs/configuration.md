# Configuración

La configuración central está en `src/bank_clients_ml/config.py`. La clase
`Settings` agrupa las rutas, los nombres de las columnas y los parámetros
globales, como `random_state` y `n_jobs`. Las variables de entorno redefinen
los valores por defecto definidos en ese archivo.

Las variables de la aplicación usan el prefijo `ML_`. Los campos anidados se
separan con `__`. Ejemplos de variables para un `.env`:

```bash
ML_DEBUG=true
ML_RANDOM_STATE=42
ML_N_JOBS=4
ML_PATHS__RAW_DATA_FILE_NAME=example_data.parquet
ML_DATASET__TARGET=Target
```

El listado completo de variables y sus valores por defecto está en `Settings`,
`DirectoryPaths`, `DatasetColumns` y `AnalysisColumns`, en
`src/bank_clients_ml/config.py`.

El archivo `.env` de la raíz del proyecto es opcional y está ignorado por Git.
El sistema solo lee del archivo las variables con prefijo `ML_`, así que las
variables de Docker Compose pueden convivir en el mismo `.env`. El contenedor
de desarrollo lee `/app/.env` mediante el montaje del proyecto; la imagen no lo
copia.

Las variables de entorno del sistema tienen prioridad sobre el `.env`, y este
sobre los valores por defecto.

En el uso normal no es necesario definir `ML_PATHS__PROJECT`: el sistema lo
calcula desde la ubicación de `config.py` y en el contenedor de desarrollo
corresponde a `/app`.

`Settings` deriva `raw_data` e `images` de `paths.project`; `raw_data` es
`paths.project / "data" / paths.raw_data_file_name`. No admiten configuración
independiente.

Al reemplazar el conjunto de datos, conservar el nombre
`example_data.parquet` o definir `ML_PATHS__RAW_DATA_FILE_NAME`. Después de
modificar el `.env` o las variables de entorno, reiniciar el proceso o el
contenedor. `Settings.settings_cached()` guarda la configuración en caché y un
proceso ya iniciado no detecta los cambios. Para invalidar esa caché, llamar a
`Settings.settings_clear_cache()`.
