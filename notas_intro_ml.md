# Notas de Estudio: Introducción al Machine Learning y Modelado

Este documento resume los conceptos fundamentales de programación, matemáticas y modelado requeridos para la introducción al Machine Learning, tomando como referencia la simulación dinámica de una pelota que rebota.

---

## 1. Fundamentos de Programación (Python)

### Estructuras de Control de Flujo

#### `if` / `else` — Toma de decisiones

El bloque `if` ejecuta un fragmento de código **solo si una condición booleana es verdadera**. El bloque `else` (opcional) se ejecuta cuando la condición es falsa.

```python
if y_siguiente <= 0:
    # La pelota tocó el suelo: aplicar rebote
    v_siguiente = -v_siguiente * e
else:
    # La pelota está en el aire: solo actúa la gravedad
    pass
```

En modelado se utilizan para establecer **condiciones de contorno**: reglas que cambian el comportamiento del sistema en situaciones especiales (un límite físico, un umbral de energía, etc.).

Los `if` pueden anidarse para representar decisiones en cascada:

```python
if y_siguiente <= 0:
    v_siguiente = -v_siguiente * e
    if v_siguiente > 0.5:   # ¿tiene energía suficiente?
        rebotes += 1
    else:
        v_siguiente = 0     # la pelota se detiene
```

#### `for` — Iteración sobre secuencias

El bucle `for` recorre **cada elemento de una secuencia** (lista, array, rango de números) y ejecuta el mismo bloque de código para cada uno.

```python
for e in e_candidatos:
    rebotes, _ = simular_rebotes(e)
    if rebotes_min <= rebotes <= rebotes_max:
        e_aceptados.append(e)
```

En simulaciones, el `for` avanza el tiempo paso a paso. En Machine Learning, es la estructura que recorre los datos de entrenamiento en cada época (`epoch`) del proceso de ajuste del modelo.

---

### Estructuras de Datos y Funciones

#### Listas

Una **lista** es una colección ordenada y mutable de elementos, definida con corchetes `[]`.

```python
trayectoria = []
trayectoria.append((0.0, 10.0))   # agregar tupla (tiempo, altura)
trayectoria.append((0.1,  9.9))
```

Las listas son la forma natural de almacenar el **historial** de un proceso iterativo: posiciones a lo largo del tiempo, pérdidas durante el entrenamiento, predicciones sobre un conjunto de datos.

Características clave:
- **Dinámicas:** crecen con `.append()` sin necesidad de declarar el tamaño de antemano.
- **Heterogéneas:** pueden contener cualquier tipo de dato (números, strings, otras listas, tuplas).
- **Indexables:** `trayectoria[0]` devuelve el primer elemento, `trayectoria[-1]` el último.

#### Funciones (`def`)

Una **función** agrupa un bloque de código reutilizable bajo un nombre. Acepta **parámetros** de entrada y puede **retornar** resultados.

```python
def simular_rebotes(e, g=9.81, y0=10.0, dt=0.01, t_max=100.0):
    """
    Retorna el número de rebotes y la trayectoria completa.
    """
    # ... lógica de la simulación ...
    return rebotes, trayectoria
```

Beneficios de usar funciones:
- **Reutilización:** el mismo código puede ejecutarse con distintos parámetros.
- **Modularidad:** cada función hace una sola cosa, lo que facilita el testing y la depuración.
- **Abstracción:** el código que llama a la función no necesita saber cómo está implementada.

En Machine Learning, el pipeline completo se organiza en funciones o clases: `cargar_datos()`, `preprocesar()`, `entrenar_modelo()`, `evaluar()`.

---

## 2. Computación Científica

### NumPy y Vectores

**NumPy** (Numerical Python) es la librería fundamental para el cálculo numérico en Python. Proporciona el tipo `ndarray` (N-dimensional array), optimizado para operaciones matemáticas sobre colecciones de datos.

#### Listas de Python vs. Arrays de NumPy

| Característica | Lista Python | Array NumPy |
|---|---|---|
| Tipo de dato | Heterogéneo | Homogéneo (más eficiente) |
| Operaciones matemáticas | Requieren `for` | Vectorizadas (sin `for`) |
| Velocidad | Lenta para cálculos numéricos | Muy rápida (C interno) |
| Uso en ML | Almacenamiento general | Datos, pesos, predicciones |

#### Operaciones vectorizadas

```python
import numpy as np

e_candidatos = np.random.uniform(0.4, 0.95, 2000)  # 2000 candidatos de una vez

# Filtrado vectorizado (sin for explícito)
condicion = (rebotes_array >= 5) & (rebotes_array <= 7)
e_aceptados = e_candidatos[condicion]
```

En Machine Learning, **todos los datos y parámetros se representan como arrays de NumPy** (o tensores en librerías como PyTorch o TensorFlow). Entender cómo operan estos arrays es un prerequisito esencial.

#### Funciones estadísticas de NumPy

```python
np.mean(array)    # media
np.var(array)     # varianza
np.std(array)     # desviación estándar
np.random.uniform(low, high, size)  # valores aleatorios uniformes
np.random.seed(42)                  # semilla para reproducibilidad
```

---

### Matplotlib

**Matplotlib** es la librería estándar para la creación de visualizaciones en Python. Se usa para:

- **Análisis exploratorio de datos:** histogramas, diagramas de dispersión, boxplots.
- **Visualización de trayectorias:** posición vs. tiempo en simulaciones físicas.
- **Monitoreo del entrenamiento:** curvas de pérdida (*loss curves*) y precisión a lo largo de las épocas.

Estructura básica de un gráfico:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))          # crear lienzo
plt.plot(x, y, color='blue')         # trazar línea
plt.scatter(x, y, alpha=0.5)         # diagrama de dispersión
plt.axhline(valor, linestyle='--')   # línea horizontal de referencia
plt.title('Título del gráfico')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 3. Fundamentos Matemáticos y Estadísticos

### Derivadas

La **derivada** de una función mide su **tasa de cambio instantánea** respecto a una de sus variables.

En física:

$$v = \frac{dy}{dt} \quad \Rightarrow \quad \text{velocidad es la derivada de la posición respecto al tiempo}$$

$$a = \frac{dv}{dt} \quad \Rightarrow \quad \text{aceleración es la derivada de la velocidad}$$

#### Integración de Euler

Como las computadoras no pueden calcular derivadas continuas de forma directa, aproximamos el movimiento con pasos discretos pequeños:

$$v_{t+\Delta t} = v_t + a \cdot \Delta t$$
$$y_{t+\Delta t} = y_t + v_{t+\Delta t} \cdot \Delta t$$

Donde $\Delta t$ (o `dt`) es el tamaño del paso de tiempo. Cuanto más pequeño sea `dt`, más precisa será la aproximación:

| `dt` | Descripción | Precisión |
|---|---|---|
| 0.5 s | Paso grande | Baja |
| 0.1 s | Paso moderado | Media |
| 0.01 s | Paso pequeño | Alta |

#### Conexión con Machine Learning

El **Gradiente Descendente** aplica la misma lógica a la optimización:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\partial L}{\partial \theta}$$

Donde:
- $\theta$ son los parámetros del modelo (equivalente a la posición $y$)
- $\alpha$ es la tasa de aprendizaje (equivalente a `dt`)
- $\frac{\partial L}{\partial \theta}$ es la derivada (gradiente) de la función de pérdida respecto a los parámetros

Al igual que Euler aproxima el movimiento continuo con pasos discretos, el Gradiente Descendente aproxima la minimización de la función de pérdida paso a paso.

---

### Estadística Descriptiva

#### Media (μ) — Valor Central

La **media aritmética** es el valor promedio de un conjunto de datos. Es la medida de **tendencia central** más común.

$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \cdots + x_n}{n}$$

En la simulación, la media de los coeficientes aceptados ($\mu_e$) es nuestra **estimación más robusta** del parámetro: el valor que, en promedio, produce el comportamiento deseado.

```python
media_e = np.mean(e_aceptados)
```

#### Varianza (σ²) — Dispersión

La **varianza** mide cuánto se alejan los datos de su media. Cuantifica la **dispersión** o **incertidumbre** en los datos.

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

Cada término $(x_i - \mu)^2$ es la **desviación al cuadrado** del elemento $i$ respecto a la media. Se usa el cuadrado para que las desviaciones positivas y negativas no se cancelen.

```python
varianza_e = np.var(e_aceptados)
```

| Varianza | Interpretación |
|---|---|
| **Alta** | Los datos están muy dispersos; alta incertidumbre en la estimación |
| **Baja** | Los datos están concentrados cerca de la media; alta certeza |

#### Desviación Estándar (σ)

La desviación estándar es la raíz cuadrada de la varianza. Tiene las mismas unidades que los datos originales, lo que la hace más interpretable:

$$\sigma = \sqrt{\sigma^2}$$

#### Conexión con Machine Learning

- **Alta varianza en predicciones** → el modelo puede estar en **overfitting** (sobreajuste): se ajustó demasiado a los datos de entrenamiento y no generaliza a nuevos datos.
- **Alta varianza en los parámetros aceptados** → existe un espectro amplio de valores del parámetro que son compatibles con el comportamiento observado. El modelo es **poco sensible** a ese parámetro (el "valley" de pérdida es plano).

---

## 4. Conceptos Base de Modelado y Machine Learning

### Modelo

Un **modelo** es una representación matemática o computacional que busca aproximar o simular un fenómeno real. Todo modelo define:

1. **Variables de entrada (inputs):** datos o condiciones que recibe el modelo.
2. **Reglas internas:** ecuaciones, lógica o parámetros que transforman los inputs.
3. **Variables de salida (outputs):** predicciones, estados, o métricas resultantes.

En nuestra simulación:
- **Input:** coeficiente de restitución `e`, condiciones iniciales
- **Regla interna:** integración de Euler con detección de colisión
- **Output:** número de rebotes, trayectoria completa

En Machine Learning:
- **Input:** datos de entrenamiento (features)
- **Regla interna:** función matemática parametrizada (ej. red neuronal)
- **Output:** predicción (clase, valor continuo, distribución de probabilidad)

### Parámetros

Los **parámetros** son las variables internas del modelo que definen su comportamiento específico.

| Tipo de modelo | Parámetros |
|---|---|
| Simulación de rebote | `e` (restitución), `g` (gravedad), `y0` (altura inicial) |
| Regresión lineal | Pendiente $w$, intercepto $b$: $\hat{y} = wx + b$ |
| Red neuronal | Pesos $W$ y sesgos $b$ de cada capa |

La diferencia clave entre una **simulación determinista** y un **modelo de ML**:
- En la simulación, los parámetros se fijan por conocimiento del dominio o se estiman mediante búsqueda.
- En ML, los parámetros se **ajustan automáticamente** desde los datos mediante un algoritmo de aprendizaje (ej. Gradiente Descendente).

### Convergencia

La **convergencia** es el estado en que un proceso iterativo se estabiliza: los valores dejan de cambiar significativamente de una iteración a la siguiente.

#### Convergencia física

En la simulación de la pelota, la convergencia ocurre cuando la energía se disipa completamente: la pelota deja de rebotar y queda en reposo ($y = 0$, $v = 0$).

#### Convergencia estadística

La varianza de los parámetros aceptados se estabiliza a medida que se agregan más experimentos. Con pocas muestras, la estimación fluctúa; con muchas, converge a un valor estable (la varianza poblacional verdadera).

#### Convergencia en Machine Learning

El entrenamiento de un modelo converge cuando el error en los datos de entrenamiento (la **función de pérdida** o *loss*) ya no disminuye de forma significativa al continuar ajustando los parámetros.

```
Época  1: loss = 2.34
Época 10: loss = 0.87
Época 50: loss = 0.21
Época 99: loss = 0.20   ← convergencia: el loss apenas cambia
Época100: loss = 0.20
```

Señales de convergencia:
- La curva de pérdida se vuelve asintóticamente plana.
- El cambio en los parámetros entre iteraciones cae por debajo de un umbral (tolerancia).
- La métrica de evaluación en el conjunto de validación deja de mejorar (*early stopping*).

---

## 5. Tabla Resumen: Simulación ↔ Machine Learning

| Concepto en la simulación | Equivalente en Machine Learning |
|---|---|
| Función `simular_rebotes(e)` | Modelo (regresión, red neuronal, etc.) |
| Coeficiente de restitución `e` | Parámetros / pesos del modelo |
| Rango objetivo [5–7 rebotes] | Función de pérdida (loss function) |
| Búsqueda aleatoria (Monte Carlo) | Optimización (ej. Gradiente Descendente) |
| Media μ de parámetros aceptados | Estimación robusta del parámetro óptimo |
| Varianza σ² de parámetros aceptados | Incertidumbre / indicador de overfitting |
| Convergencia de la varianza | Convergencia del entrenamiento |
| `dt` (paso de tiempo en Euler) | Tasa de aprendizaje $\alpha$ en Gradiente Descendente |

---

*Material de apoyo para el Módulo 1 — Introducción al Machine Learning a través del Modelado Computacional.*
