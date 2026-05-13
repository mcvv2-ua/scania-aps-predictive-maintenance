# Prediccion explicable y sensible al coste de fallos APS en camiones Scania

Este proyecto usa el dataset **APS Failure at Scania Trucks** del UCI Machine Learning Repository.
El objetivo es detectar si un fallo de un camion esta relacionado con el sistema APS.

La tarea es una clasificacion binaria.

- Clase 0. Fallo no relacionado con APS.
- Clase 1. Fallo relacionado con APS.

La clase positiva es la mas importante.
Un falso negativo puede dejar pasar un fallo APS real.
Por eso accuracy no basta.

## Dataset

El dataset es tabular.
Contiene variables anonimizadas de sensores y contadores.
Tiene muchos valores faltantes.
La clase positiva esta muy desbalanceada.

El coste usado es el indicado por el dataset.

- Falso positivo. 10.
- Falso negativo. 500.

## Estructura

```text
proyecto_scania/
  README.md
  requirements.txt
  .gitignore
  project_utils.py
  00_descarga_y_preparacion.ipynb
  01_EDA.ipynb
  02_modelado_supervisado.ipynb
  03_desbalanceo_y_coste.ipynb
  04_anomalias.ipynb
  05_xai_robustez_errores.ipynb
  06_resultados_memoria_y_defensa.ipynb
  data/
    raw/
    processed/
  artifacts/
    metrics/
    models/
    figures/
    tables/
```

## Instalacion

Entra en la carpeta donde este el proyecto.
Ejecuta los comandos desde la raiz de `proyecto_scania`.

Windows:

```bash
cd ruta\a\proyecto_scania
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux o macOS:

```bash
cd ruta/a/proyecto_scania
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

Abre Jupyter desde la raiz del proyecto.

```bash
jupyter notebook
```

Ejecuta los notebooks en este orden.

1. `00_descarga_y_preparacion.ipynb`
2. `01_EDA.ipynb`
3. `02_modelado_supervisado.ipynb`
4. `03_desbalanceo_y_coste.ipynb`
5. `04_anomalias.ipynb`
6. `05_xai_robustez_errores.ipynb`
7. `06_resultados_memoria_y_defensa.ipynb`

## Protocolo de evaluacion

El test se separa al principio.
Test queda reservado para la evaluacion final.
No se usa para elegir modelos.
No se usa para elegir estrategias.
No se usa para ajustar hiperparametros.
No se usa para ajustar umbrales.
No se usa para tomar decisiones durante el entrenamiento.

Cada modelo supervisado usa `Pipeline`.
Cada modelo supervisado usa `GridSearchCV` con 5-fold CV solo sobre train.
La mejor configuracion de cada modelo se selecciona solo con train.
El modelo principal se selecciona solo con resultados de CV en train.
HistGradientBoostingClassifier es una implementacion de Gradient Boosting en sklearn y se evalua con el mismo protocolo.
Despues se reentrena con todo train mediante `GridSearchCV(refit=...)`.
Los resultados de test se usan solo para la comparativa final.

El analisis de desbalanceo se centra en pesos de clase, coste asimetrico y metricas adecuadas.
Esto evita data leakage.

## Analisis posterior

Los notebooks de anomalias, XAI, robustez, errores y shift se ejecutan despues de fijar el modelo.
Son auditoria final.
No modifican el modelo.
No modifican los hiperparametros.
No modifican el umbral.
No cambian decisiones de entrenamiento.

## Metricas

Se reportan metricas adecuadas para desbalanceo.

- Accuracy.
- Balanced accuracy.
- Precision de la clase positiva.
- Recall de la clase positiva.
- F1 de la clase positiva.
- ROC-AUC.
- PR-AUC.
- Matriz de confusion.
- Coste total.

La comparativa debe mirar especialmente recall, F1, PR-AUC y coste.

## Artefactos

Los resultados se generan al ejecutar los notebooks.
No hay resultados inventados.

Se generan artefactos en `artifacts/`.

- Resultados de CV.
- Resultados finales en test.
- Parametros seleccionados.
- Modelo principal.
- Matrices de confusion.
- Tablas de coste.
- Resultados de anomalias.
- Resultados de robustez.
- Analisis de errores.
- Analisis de shift.
- Importancia de variables.
- Tablas preparadas para memoria.

## Tiempos

El dataset es grande.
Algunos modelos pueden tardar.
Los grids son pequenos para que el proyecto sea ejecutable en un ordenador normal.
One-Class SVM se entrena con una muestra para evitar tiempos excesivos.

## Archivos que no se deben subir

No subas `.venv/`.
No subas `__pycache__/`.
No subas checkpoints de notebooks.
No subas datos raw si se descargan automaticamente.
No subas modelos pesados si no son necesarios para la entrega.
