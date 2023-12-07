# Trabajo Practico Integrador - Aprendizaje Automático 1

## Tecnicatura Universitaria en Inteligencia Artificial (TUIA)

### **Docentes**

- Spak, Joel
- Agustín Almada
- Bruno Cocitto

### **Integrantes**

| Apellido, Nombre | Legajo |
| --- | --- |
| Aguirre, Fabian | A-4516/1 |
| Fontela, Facundo  | F-3724/9 |

**Año**: 2023

---


En el presente trabajo, se exploró el dataset weatherAUS, correspondiente a registros de lluvia de distintas ciudades de Australia, con el objetivo de aplicar técnicas y modelos de regresión y clasificación para poder predecir si, a partir de un registro particular, al día siguiente lloverá y qué cantidad.

El trabajo comenzó con un análisis exploratorio del dataset e ingeniería de características, donde se ajustaron datos faltantes, nulos, etc. Se filtró las distintas ciudades e innecesarias y se redujo el dataset a una versión apta para aplicar los modelos correspondientes.
Es de destacar que se seleccionaron 15 columnas (features), para trabajar y aplicar los modelos, a partir de un criterio de correlación lineal. Esta decisión es un criterio particular, pero no así el ideal. En otra posibilidad sería mejor utilizar todas las características y hacer una selección median Ridge y Lasso.

En el trabajo se aplicaron distintos modelos:

  * Regresión lineal: donde se obtuvo métricas bastante pobres, r2=0.2.
  * Gradiente descendiente y Gradiente descendiente estocástico para analizar el comportamiento de MSE, no se encontró una mejoría significativa en contraste a la regresión lineal.
   
Métodos de regularización, obteniendo una leve mejoría en las métricas (r2=0.22)
  * Ridge
  * Lasso
  * Elastic Net

Modelos de clasificación, donde nos centramos en la métrica de recall, dado que el dataset no se encuentra balanceado:
  * Regresión logística (sin balancear): recall=(0.95-0.5)
  * Estrategias de balanceo utilizadas: oversampler, undersampler, near-miss: Se obtuvo una mejora en la prediccion de los dias de lluvia a costa de una disminucion en el recall de los dias que no llueve. recall=(0.79-0.76).

Implementacion de modelos base:
  * Regresión: un modelo que predice siempre 0, ya que la mayor cantidad de días no llueve.
  * Clasificación: modelo que predice siempre "No", por el mismo argumento.

Implementacion de Redes neuronales:
 * Regresión: Red con 2 capas ocultas de 18 y 9 neuronas respectivamete y salida lineal. Pudiendo elevar r2 hasta valores de 0.35.
 * Clasificación: Red con 2 capas ocultas de 19 y 10 neuronas respectivamete y salida mediante una función sigmoidea. Se obtuvo una mejoría en la métrica de interés recall=(0.79-0.80)

Tunning de hiper paramétros mediante OPTUNA para ambas redes neuronales. No se observó notables mejorías en las métricas.
Sería de interés explicitar que el valor obtenido como "óptimo" para el batch_size empeoró significativamente el modelo. Por lo que se decidió no hacer caso a este y optar por uno empírico.

Gráficos de explicabilidad con SHAP.

Como conclusión general: los modelos de regresión desarrollados son bastante pobres, con métricas que dejan bastante que desear, a pesar de las distintas técnicas empleadas. Sería de ayuda contar con otras caracteristicas y mayor cantidad de datos para poder profundizar el análisis. Por otro lado, los modelos de clasificación tienen un porcentaje de acierto elevado, siendo bastante satisfactorio los resultados obtenidos.
