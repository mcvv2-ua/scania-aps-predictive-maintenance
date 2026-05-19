# Predicción de fallos APS en camiones Scania

Proyecto de **aprendizaje automático para mantenimiento predictivo** sobre datos tabulares del sistema APS de camiones Scania. El trabajo aborda un problema con **clase positiva muy minoritaria**, valores faltantes, coste asimétrico de los errores, análisis de explicabilidad, robustez, errores, anomalías y shift entre train y test.

**Autores:** Mauro Valls Vidal, Alejandro Parra Sánchez, Jordi Blasco Lozano y Alejandro Martínez Riquelme.  
**Asignatura:** Aprendizaje Avanzado — Grado en Ingeniería en Inteligencia Artificial, Universidad de Alicante (Curso 2025–2026).

## Objetivo

El objetivo es detectar fallos relacionados con el sistema **APS** y reducir el riesgo de **falsos negativos**, ya que dejar pasar un fallo real puede tener un impacto operativo mayor que revisar un camión que finalmente no presenta ese fallo.

La tarea se formula como clasificación binaria.

| Clase       | Significado                  |
| ----------- | ---------------------------- |
| `0` / `neg` | Fallo no relacionado con APS |
| `1` / `pos` | Fallo relacionado con APS    |

## Dataset

Se utiliza el dataset **APS Failure at Scania Trucks** del UCI Machine Learning Repository.

| Partición | Filas |
| --------- | ----: |
| Train     | 60000 |
| Test      | 16000 |

El dataset contiene **170 variables predictoras** anonimizadas procedentes de sensores, contadores o agregados internos del camión. Esta anonimización supone un reto porque limita la interpretación física directa de cada variable y obliga a centrar el análisis en patrones estadísticos, rendimiento predictivo y explicabilidad indirecta. La clase positiva es minoritaria y existen numerosos valores faltantes, por lo que accuracy no es suficiente como métrica principal.

## Protocolo experimental

El proyecto sigue un protocolo pensado para evitar fuga de información.

- La selección de modelos y configuraciones se realiza usando solo **train**.
- Los modelos supervisados se entrenan mediante **Pipeline**.
- La búsqueda de hiperparámetros se realiza con **GridSearchCV**.
- Se usa **validación cruzada estratificada de 5 folds**.
- La métrica principal de selección es **PR-AUC / average precision**, adecuada para clase positiva minoritaria.
- El conjunto **test** se reserva exclusivamente para la evaluación final.
- No se ajustan hiperparámetros ni umbrales usando test.

## Modelos comparados

| Familia   | Modelos                                                    |
| --------- | ---------------------------------------------------------- |
| Baseline  | Dummy classifier                                           |
| Lineales  | Logistic Regression, Linear SVM                            |
| Árboles   | Decision Tree                                              |
| Ensembles | Random Forest, Extra Trees, AdaBoost, HistGradientBoosting |

Todos los modelos se comparan bajo el mismo protocolo de validación. El baseline sirve como referencia para comprobar que los modelos aprenden señal real y no solo explotan el desbalanceo.

## Resultados principales

El modelo principal seleccionado es **HistGradientBoosting**, elegido exclusivamente por validación cruzada en train al obtener la mejor **PR-AUC** media.

| Resultado                  | Valor aproximado |
| -------------------------- | ---------------: |
| CV PR-AUC                  |            0.881 |
| Test PR-AUC                |            0.922 |
| Test F1 clase positiva     |            0.842 |
| Test recall clase positiva |            0.752 |
| Falsos positivos en test   |               13 |
| Falsos negativos en test   |               93 |

La matriz de confusión del modelo principal en test es la siguiente.

|               | Pred. negativo | Pred. positivo |
| ------------- | -------------: | -------------: |
| Real negativo |          15612 |             13 |
| Real positivo |             93 |            282 |

Estos resultados hacen que **HistGradientBoosting** sea el modelo principal defendible del proyecto. Mantiene buen ranking de positivos, buen F1 de la clase positiva y una mejora clara frente al baseline, sin usar test durante la selección.

## Análisis complementarios

Además del modelado supervisado principal, el proyecto incluye análisis posteriores para interpretar mejor el comportamiento del sistema.

- **Coste asimétrico:** Se considera un coste de 10 para falsos positivos y 500 para falsos negativos.
- **Estrategia sensible al coste:** `logistic_class_weight` reduce muchos falsos negativos y obtiene menor coste total, por lo que es una alternativa si el criterio prioritario fuese coste. No se presenta como modelo final.
- **Anomalías:** Se estudian detectores de anomalías como análisis secundario. No sustituyen al modelo supervisado.
- **XAI:** Se utiliza importancia por permutación para identificar variables influyentes del modelo.
- **Robustez:** Se evalúa la degradación del rendimiento ante ruido.
- **Análisis de errores:** Se revisan falsos positivos y falsos negativos para entender patrones de fallo.
- **Shift train-test:** Se analiza si existen diferencias entre las distribuciones de train y test.

## Estructura del repositorio

```text
proyecto_scania/
  README.md
  requirements.txt
  project_utils.py
  memoria_scania_aps.pdf
  00_descarga_y_preparacion.ipynb
  01_EDA.ipynb
  02_modelado_supervisado.ipynb
  03_desbalanceo_y_coste.ipynb
  04_anomalias.ipynb
  05_xai_robustez_errores.ipynb
  data/        # generada localmente si no se sube completa al repositorio
  artifacts/   # generada localmente si no se sube completa al repositorio
```

| Notebook                          | Contenido                                                                                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `00_descarga_y_preparacion.ipynb` | Descarga o carga del dataset, limpieza inicial, conversión de etiquetas y guardado de train/test procesados.                                             |
| `01_EDA.ipynb`                    | Análisis exploratorio de distribución de clases, missing values, escalas, outliers, correlaciones y métricas adecuadas.                                  |
| `02_modelado_supervisado.ipynb`   | Comparación principal de modelos con `Pipeline`, `GridSearchCV` y 5-fold CV. Incluye la selección de HistGradientBoosting y la evaluación final en test. |
| `03_desbalanceo_y_coste.ipynb`    | Estudio del desbalanceo y del coste asimétrico. Incluye la comparación con `logistic_class_weight` como alternativa si se prioriza coste.                |
| `04_anomalias.ipynb`              | Análisis secundario de detección de anomalías para estudiar si los fallos APS se comportan como casos raros.                                             |
| `05_xai_robustez_errores.ipynb`   | Explicabilidad con importancia por permutación, robustez ante ruido, análisis de errores y shift train-test.                                             |

## Instalación

Desde la raíz del proyecto.

```bash
python -m venv .venv
```

En Windows.

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux o macOS.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

También puede ejecutarse en un entorno conda equivalente instalando las dependencias desde `requirements.txt`.

## Orden de ejecución

Ejecutar los notebooks en este orden.

1. `00_descarga_y_preparacion.ipynb`
2. `01_EDA.ipynb`
3. `02_modelado_supervisado.ipynb`
4. `03_desbalanceo_y_coste.ipynb`
5. `04_anomalias.ipynb`
6. `05_xai_robustez_errores.ipynb`

## Limitaciones

- Las variables están **anonimizadas**, lo que limita la interpretación física de los resultados.
- La clase positiva es **muy minoritaria**, por lo que pequeñas variaciones en falsos negativos afectan mucho a recall y coste.
- Hay muchos **valores faltantes**, tratados mediante imputación con mediana dentro de los pipelines, excepto en HistGradientBoosting, que los gestiona de forma nativa.
- El coste usado es una **simplificación** basada en la penalización del dataset.
- El proyecto no es un despliegue real en producción. No incluye monitorización, integración con sistemas de taller ni validación operacional continua.

## Conclusión

El proyecto muestra que **HistGradientBoosting** es una elección principal sólida para la detección de fallos APS. El modelo fue seleccionado mediante **validación cruzada en train** usando **PR-AUC** como métrica principal, y después mantuvo un rendimiento alto en test sin usar esta partición durante la selección.

La mejora frente al baseline confirma que los modelos aprenden señal útil en las variables del sistema, a pesar del fuerte desbalanceo, los valores faltantes y la complejidad de los datos industriales. Aun así, el análisis de errores muestra que todavía existen falsos negativos relevantes, por lo que el modelo no debe interpretarse como una solución perfecta.

Los análisis complementarios aportan una lectura más realista del problema. El estudio de coste muestra que **logistic_class_weight** puede ser una alternativa interesante si el objetivo operativo principal es reducir fallos no detectados, aunque genere más falsas alarmas. Los análisis de anomalías, explicabilidad, robustez y shift ayudan a entender mejor las ventajas, limitaciones y posibles usos del sistema.

En conjunto, el proyecto no solo compara modelos, sino que construye un flujo experimental completo y defendible para un problema real de mantenimiento predictivo.
