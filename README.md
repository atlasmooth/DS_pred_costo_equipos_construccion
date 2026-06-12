# Informe de Consultoría — Predicción de Costos Operativos

**Proyecto:** Predicción de costos operativos en proyecto de construcción  
**Consultor:** Dennis Andres Forero — Data Scientist    
**Corte de análisis:** 2023-08-31

---

## Contexto y Motivación

Las empresas del sector de construcción e infraestructura enfrentan un reto estructural de planeación: definir un presupuesto de costos operativos que sea **certero, realista y que incorpore la incertidumbre inherente al futuro**. La dependencia de insumos cuyos precios fluctúan en el tiempo convierte la planeación presupuestal en un ejercicio de pronóstico, donde decisiones tomadas hoy tienen impacto directo en la continuidad operativa del mañana.

Este repositorio documenta una solución analítica completa que aborda este problema usando **modelos estadísticos y de aprendizaje automático sobre series de tiempo**, con el objetivo de predecir el precio futuro de dos equipos operativos (`Equipo1` y `Equipo2`) a partir de información histórica y de insumos correlacionados (`X`, `Y`, `Z`).

---

## Estructura del Repositorio

```
.
├── ds_cost_prediction.ipynb              # Notebook principal: EDA, modelado, resultados
├── decomposer.py                         # Clase auxiliar para análisis y descomposición de series de tiempo
├── data/
│   ├── historico_equipos.csv             # Serie histórica diaria de precios de equipos e insumos
│   ├── X.csv                             # Fuente externa: precio histórico del insumo X
│   ├── Y.csv                             # Fuente externa: precio histórico del insumo Y
│   └── Z.csv                             # Fuente externa: precio histórico del insumo Z
└── Informe de Consultoría.pdf   # Informe ejecutivo del caso
```

---

## Datos

El conjunto de datos principal (`historico_equipos.csv`) contiene **3,530 registros** de frecuencia diaria entre el 2010-01-04 y el 2023-08-31, con las columnas:

| Variable       | Descripción                                  |
|----------------|----------------------------------------------|
| `Price_X`      | Precio del insumo X (fuente interna)         |
| `Price_Y`      | Precio del insumo Y (fuente interna)         |
| `Price_Z`      | Precio del insumo Z (fuente interna)         |
| `Price_Equipo1`| **Target:** precio del equipo 1              |
| `Price_Equipo2`| **Target:** precio del equipo 2              |

Se identificaron saltos temporales de hasta 4 días sin registros. Estos fueron imputados mediante **interpolación lineal** entre el último y el siguiente valor conocido, bajo el supuesto de que el precio varía de forma continua dentro del intervalo sin observaciones.

Las fuentes externas `X.csv`, `Y.csv`, `Z.csv` fueron validadas contra el histórico consolidado, confirmando consistencia de valores donde coinciden las fechas.

---

## Supuestos Clave

1. La **fecha de corte presente** es el 2023-08-31. No se usa ningún dato posterior a esta fecha para el entrenamiento o la generación del pronóstico.
2. El precio de un equipo **no puede usarse como feature** para predecir el otro.
3. Los vacíos de hasta 4 días se imputan por interpolación lineal.
4. El horizonte de predicción es **configurable** (parámetro `future_horizon`, por defecto 30 días), y la entrega del pronóstico es en frecuencia **diaria**.
5. Los precios futuros de los insumos X, Y, Z **no están disponibles** para el período de pronóstico; solo se usan sus valores lageados (pasados) como features.

---

## Metodología

### Análisis Exploratorio (EDA)

El EDA cubre tres áreas:

- **Estadística descriptiva y calidad de datos:** métricas de tendencia central, dispersión, tipos y nulos, frecuencia inferida, gaps temporales.
- **Análisis de distribuciones:** histogramas, boxplots, violin plots, Q-Q plots, asimetría (skewness) y curtosis para cada variable.
- **Descomposición estacional:** se aplican STL y MSTL sobre cada serie para identificar componentes de tendencia, estacionalidad y residuo. Se confirma un **ciclo anual significativo** en todas las variables.

### Análisis de Correlación

Los scatter plots y matrices de correlación revelan:
- `Price_Equipo1` presenta **alta colinealidad positiva** con `Price_Y`.
- `Price_Equipo2` presenta **alta colinealidad positiva** con `Price_Z`.

Estas relaciones guían la selección de features y sugieren que los precios de los insumos son los predictores más informativos, sin descartar efectos de los demás.

### Feature Engineering

Se generan automáticamente en función del parámetro `future_horizon`:

| Tipo de feature               | Detalle                                            |
|------------------------------|----------------------------------------------------|
| Temporales                   | `year`, `month`, `dayofweek`, `dayofyear`          |
| Lags del target              | t-1, t-7, t-30, t-365 (con recursión del modelo)  |
| Lags de insumos              | Valores de `Price_Y` y `Price_Z` en t-365          |
| Componente estacional (STL)  | Valor del ciclo anual para el día a predecir       |

Los modelos basados en árboles no requieren escalado de features ni transformación de targets.

### Selección y Evaluación de Modelos

Se comparan los siguientes modelos en una estrategia de **cross-validation en series de tiempo** (20 ventanas, paso = `future_horizon`), todos configurados con la misma arquitectura de features:

- `LGBMRegressor`
- `CatBoostRegressor`
- `XGBRegressor`

La métrica de optimización por defecto es **SMAPE** (Symmetric Mean Absolute Percentage Error). El mejor modelo se elige por promedio de ranking de error en todas las ventanas y targets.

**Resultados para `future_horizon = 30`:**

| Métrica | Valor           |
|---------|-----------------|
| SMAPE   | 2.41 %          |
| MAPE    | 4.80 %          |
| RMSE    | 47.24           |
| MAE     | 40.60           |

> El modelo ganador fue **XGBRegressor**.

### Pronóstico

Con el modelo seleccionado se realiza un re-entrenamiento sobre toda la historia disponible (`history_augmented`) y se genera el pronóstico para los próximos `future_horizon` días, incluyendo intervalos de confianza al 95% mediante `PredictionIntervals` (conformal prediction, 50 ventanas).

---

## Decisión de Enfoque

Se evaluaron dos alternativas:

- **Alternativa A:** modelos estadísticos o regresión lineal por separado para cada target, usando features lageados del insumo de mayor correlación.
- **Alternativa B (seleccionada):** modelo global de regresión basado en árboles, entrenado con datos de ambos targets, con lags de insumos, características temporales y componentes estacionales.

La opción B fue elegida porque los modelos basados en árboles capturan relaciones **no lineales** que el EDA evidencia más allá de la alta colinealidad, y porque la estrategia global permite aprovechar patrones compartidos entre los dos equipos.

---

## Clase Auxiliar: `Decomposer`

El archivo [decomposer.py](decomposer.py) contiene la clase `Decomposer`, diseñada para el análisis estadístico y la descomposición de series de tiempo, tanto individuales como catálogos de múltiples series.

### Capacidades principales

| Método / Grupo                          | Descripción                                                                                           |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Stationarity (ADF)**                  | Calcula el estadístico ADF y el p-value para determinar si una serie es estacionaria.                |
| **Classic Decomposition**               | Descomposición aditiva/multiplicativa clásica vía `seasonal_decompose` de statsmodels.               |
| **STL Decomposition**                   | Descomposición robusta usando STL (Seasonal-Trend decomposition via LOESS), configurable por frecuencia y estacionalidad. |
| **MSTL Decomposition**                  | Descomposición con múltiples estacionalidades simultáneas (e.g., semanal + anual en datos diarios).  |
| **Catalog methods**                     | Versiones paralelas de cada descomposición sobre un DataFrame con múltiples series (via `fugue`).    |
| **Plot methods**                        | Funciones de visualización para cada tipo de descomposición.                                         |

### Uso rápido

```python
import decomposer

# Descomposición STL anual sobre serie diaria
result = decomposer.Decomposer.series_stl_decomposition(
    t_series=my_series,
    data_freq='D',
    seasonal_freq='Y'
)

# Gráfico de descomposición MSTL con estacionalidades semanal, mensual y anual
decomposer.Decomposer.mstl_decomposition_plot(my_series, 'D', ['W', 'M', 'Y'])

# Test de estacionariedad
print(decomposer.Decomposer.stationarity(my_series))  # 'stationary' | 'non-stationary'
```

La clase puede usarse tanto de forma **estática** (sin instanciar) como de forma **instanciada** para evitar recalcular métricas repetidamente.

---

## Mejoras Futuras

- **Hiperparametrización automática** con Optuna para cada modelo candidato.
- Incorporar **transformaciones adicionales de features** (tendencia, FFT) sin caer en alta dimensionalidad.
- Evaluar modelos **dedicados por equipo** vs. modelo global.
- Estrategia de **descomposición + pronóstico por componente** (trend, seasonal, residuo por separado).
- Ponderación de ventanas por **proximidad temporal** (Moving Average) en la selección de modelo.
- Exploración de modelos de **deep learning** y modelos fundacionales preentrenados para secuencias (e.g., basados en Transformer).

---

## Dependencias Principales

| Librería            | Uso                                               |
|---------------------|---------------------------------------------------|
| `mlforecast`        | Framework de forecasting con ML, cross-validation |
| `lightgbm`          | Modelo LGBMRegressor                              |
| `catboost`          | Modelo CatBoostRegressor                          |
| `xgboost`           | Modelo XGBRegressor                               |
| `statsmodels`       | STL, MSTL, descomposición clásica, ADF test       |
| `utilsforecast`     | Métricas de evaluación, plotting                  |
| `fugue`             | Procesamiento paralelo para catálogos de series   |
| `pandas` / `numpy`  | Manipulación de datos                             |
| `matplotlib` / `seaborn` | Visualización                               |

---

> El notebook [`ds_cost_prediction.ipynb`](ds_cost_prediction.ipynb) contiene la implementación completa, reproducible paso a paso, desde la carga de datos hasta el pronóstico final con intervalos de confianza.
