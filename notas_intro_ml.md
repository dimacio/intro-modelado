# Notas de Estudio: Introducción al Machine Learning y Modelado

Este documento es el material de lectura complementario del notebook `intro_ml_pedagogico.ipynb`. Cada sección incluye una referencia a la parte del notebook donde se practica el concepto.

---

## 1. Fundamentos de Programación (Python)

> 🖥️ **Práctica en el notebook:** Parte 1 (celdas 1.1 a 1.4)

### `if` / `else` — Toma de decisiones

El bloque `if` ejecuta código **solo si una condición booleana es verdadera**. El bloque `else` (opcional) se ejecuta cuando la condición es falsa.

```python
if y_siguiente <= 0:
    v_siguiente = -v_siguiente * e   # rebote
else:
    pass                             # la gravedad actúa normalmente
```

Los `if` pueden anidarse para decisiones en cascada. En la simulación:
1. El `if` externo detecta si la pelota tocó el suelo.
2. El `if` interno decide si tiene suficiente energía para rebotar o debe detenerse.

**En ML:** los `if` implementan condiciones de corte (convergencia, early stopping), umbrales de clasificación, y cualquier lógica condicional dentro del pipeline.

---

### `for` — Iteración sobre secuencias

El bucle `for` recorre cada elemento de una secuencia (lista, array, rango de números) y ejecuta el mismo bloque de código para cada uno.

```python
for e in e_candidatos:
    rebotes, _ = simular_rebotes(e)
    if rebotes_min <= rebotes <= rebotes_max:
        e_aceptados.append(e)
```

**En ML:** el `for` es la estructura que recorre los datos de entrenamiento en cada época del proceso de ajuste del modelo (el bucle de entrenamiento).

---

### Listas

Una **lista** es una colección ordenada y mutable de elementos, definida con corchetes `[]`. Se amplía con `.append()` sin necesidad de declarar tamaño.

```python
trayectoria = []
trayectoria.append((0.0, 10.0))   # (tiempo, altura)
trayectoria.append((0.1,  9.9))
```

**En ML:** las listas almacenan el historial de pérdidas durante el entrenamiento, las predicciones sobre un conjunto de test, o los resultados de cada experimento.

---

### Funciones (`def`)

Una función agrupa código reutilizable bajo un nombre, acepta parámetros y retorna resultados.

```python
def simular_rebotes(e, g=9.81, y0=10.0, dt=0.01, t_max=100.0):
    # ...
    return rebotes, trayectoria
```

Los parámetros con valor por defecto (como `g=9.81`) son opcionales: si no se pasan, toman el valor predefinido. Esto permite llamar a la función con solo el parámetro que varía (`simular_rebotes(0.75)`) y mantener el resto constante.

**En ML:** el pipeline completo se organiza en funciones o clases: `cargar_datos()`, `preprocesar()`, `entrenar()`, `evaluar()`.

---

## 2. Computación Científica

> 🖥️ **Práctica en el notebook:** Parte 2 (celdas 2.1 y 2.2)

### NumPy y Vectores

**NumPy** provee el tipo `ndarray`: un arreglo de datos del mismo tipo, optimizado para operaciones matemáticas. La diferencia clave con una lista de Python:

| Característica | Lista Python | Array NumPy |
|---|---|---|
| Tipo de dato | Heterogéneo | Homogéneo (más eficiente) |
| Operaciones matemáticas | Requieren `for` | Vectorizadas (sin `for`) |
| Velocidad numérica | Lenta | Muy rápida (C interno) |
| Uso principal en ML | Almacenamiento general | Datos, pesos, predicciones |

#### Operaciones vectorizadas

```python
# Lista: necesito un for para operar sobre cada elemento
lista_doble = [x * 2 for x in lista]

# Array: la operación se aplica a todos los elementos a la vez
array_doble = array * 2
```

#### Filtrado vectorial (máscara booleana)

```python
condicion   = (rebotes_array >= 5) & (rebotes_array <= 7)  # array de True/False
e_aceptados = e_candidatos[condicion]                       # selección directa
```

Esta forma de filtrar es más eficiente y legible que un `for` con `if`.

#### Funciones estadísticas clave

```python
np.mean(array)                       # media
np.var(array)                        # varianza
np.std(array)                        # desviación estándar
np.random.uniform(low, high, size)   # muestras uniformes aleatorias
np.random.seed(42)                   # semilla para reproducibilidad
```

**En ML:** todos los datos (features), parámetros del modelo y predicciones se representan como arrays de NumPy (o tensores, que son su equivalente en PyTorch/TensorFlow).

---

### Matplotlib

Matplotlib crea gráficos estáticos, animados e interactivos.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(x, y, color='blue', linewidth=2, label='Trayectoria')
plt.scatter(x, y, alpha=0.5, s=12)
plt.axhline(valor, linestyle='--', label='Referencia')
plt.title('Título')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**En ML:** se usa para el análisis exploratorio de datos, para graficar la curva de pérdida durante el entrenamiento, y para visualizar distribuciones de predicciones.

---

## 3. Fundamentos Matemáticos y Estadísticos

> 🖥️ **Práctica en el notebook:** Parte 3 (celdas 3.1 y 3.2)

### Derivadas

La **derivada** de una función mide su **tasa de cambio instantánea** respecto a una de sus variables.

En física:
$$v = \frac{dy}{dt} \qquad \text{(velocidad = derivada de la posición)}$$
$$a = \frac{dv}{dt} = -g \qquad \text{(aceleración = derivada de la velocidad)}$$

#### Integración de Euler

Como las computadoras trabajan con pasos discretos, aproximamos las derivadas:

$$v_{t+\Delta t} = v_t - g \cdot \Delta t$$
$$y_{t+\Delta t} = y_t + v_{t+\Delta t} \cdot \Delta t$$

Cuanto más pequeño es $\Delta t$, más precisa es la aproximación (ver experimento en el notebook). El costo es mayor tiempo de cómputo.

#### Conexión con Machine Learning: Gradiente Descendente

El Gradiente Descendente aplica exactamente la misma lógica a la optimización de parámetros:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\partial L}{\partial \theta}$$

| Euler (simulación) | Gradiente Descendente (ML) |
|---|---|
| Posición $y$ | Parámetros del modelo $\theta$ |
| Velocidad $v$ | Gradiente $\frac{\partial L}{\partial \theta}$ |
| Paso de tiempo $\Delta t$ | Tasa de aprendizaje $\alpha$ |
| Aceleración $-g$ | Dirección de descenso del gradiente |

---

### Estadística Descriptiva

#### Media (μ) — Valor central

$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

En la simulación, la media de los $e$ aceptados es nuestra estimación más robusta del parámetro: el valor que, en promedio, produce el comportamiento deseado.

#### Varianza (σ²) — Dispersión

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

Cada término $(x_i - \mu)^2$ es la desviación al cuadrado del dato $i$ respecto a la media. Se usa el cuadrado para que las desviaciones positivas y negativas no se cancelen.

```python
# Implementación explícita (equivalente a np.var)
media        = np.mean(datos)
desviaciones = datos - media          # distancia de cada punto a la media
varianza     = np.mean(desviaciones**2)
```

| Varianza | Interpretación |
|---|---|
| Alta | Datos muy dispersos; alta incertidumbre en la estimación |
| Baja | Datos concentrados cerca de la media; alta certeza |

**En ML:** varianza alta en las predicciones de un modelo puede indicar *overfitting*: el modelo se ajustó demasiado a los datos de entrenamiento y no generaliza a datos nuevos.

---

## 4. Conceptos Base de Modelado y Machine Learning

> 🖥️ **Práctica en el notebook:** Partes 4 y 5

### Modelo

Un **modelo** es una representación matemática o computacional que busca aproximar un fenómeno real. Todo modelo define:

1. **Inputs:** datos o condiciones de entrada.
2. **Reglas internas:** ecuaciones o lógica que transforma los inputs (los parámetros determinan el comportamiento).
3. **Outputs:** predicciones o estados resultantes.

En nuestra simulación:

| Componente | Valor |
|---|---|
| Input | Coeficiente de restitución `e`, condiciones iniciales |
| Regla interna | Integración de Euler + detección de colisión |
| Output | Número de rebotes, trayectoria completa |

---

### Parámetros

Los **parámetros** son las variables internas del modelo que definen su comportamiento específico.

| Modelo | Parámetros |
|---|---|
| Simulación de rebote | `e`, `g`, `y0`, `dt` |
| Regresión lineal | Pendiente $w$, intercepto $b$ |
| Red neuronal | Pesos $W$ y sesgos $b$ de cada capa |

La diferencia clave entre simulación y ML:
- En la simulación, los parámetros se fijan por conocimiento del dominio o se estiman por búsqueda.
- En ML, los parámetros se **ajustan automáticamente** a partir de los datos mediante un algoritmo de optimización.

---

### Muestreo por rechazo vs. Integración Monte Carlo

> 🖥️ **Práctica en el notebook:** Parte 6

Ambos métodos usan muestreo aleatorio masivo pero responden preguntas distintas:

| | Muestreo por rechazo | Integración Monte Carlo |
|---|---|---|
| **Pregunta** | ¿Cuáles son los $e$ aceptados? | ¿Cuánto vale esta integral sobre $e$? |
| **Output** | Una distribución de parámetros | Un número (probabilidad, media, etc.) |
| **Operación** | Filtrar y guardar candidatos | Promediar una función sobre muestras |

**La simulación es idéntica en ambos casos.** Lo que cambia es cómo se procesan los resultados.

La media de los $e$ aceptados puede expresarse como integral Monte Carlo:

$$\mu_e = \frac{\displaystyle\frac{1}{N}\sum_i e_i \cdot \mathbf{1}_i}{\displaystyle\frac{1}{N}\sum_i \mathbf{1}_i} = \frac{\text{numerador}}{\text{denominador}}$$

donde $\mathbf{1}_i = 1$ si el candidato $i$ es aceptado, $0$ si no. Los $\frac{1}{N}$ se cancelan y el resultado es idéntico al promedio de los candidatos filtrados: el rechazo es un caso especial de la integración Monte Carlo.

---

### Convergencia

La **convergencia** es el estado en que un proceso iterativo se estabiliza.

**Convergencia física:** la pelota deja de rebotar cuando toda su energía cinética se disipa ($v = 0$, $y = 0$).

**Convergencia estadística:** la varianza de los parámetros aceptados se estabiliza a medida que se agregan más experimentos. Con pocas muestras fluctúa; con muchas converge a la varianza poblacional verdadera.

**Convergencia en ML:** el entrenamiento converge cuando la función de pérdida ya no disminuye de forma significativa:

```
Época  1: loss = 2.34
Época 10: loss = 0.87
Época 50: loss = 0.21
Época 99: loss = 0.20   ← convergencia
Época100: loss = 0.20
```

Señales de convergencia: la curva de pérdida se vuelve asintóticamente plana, el cambio en los parámetros cae por debajo de una tolerancia, o la métrica en validación deja de mejorar (*early stopping*).

---

## 5. Tabla Resumen: Simulación ↔ Machine Learning

> 🖥️ **Ver al final del notebook:** sección Resumen

| Concepto en la simulación | Equivalente en Machine Learning |
|---|---|
| Función `simular_rebotes(e)` | Modelo (red neuronal, regresión, etc.) |
| Coeficiente de restitución `e` | Parámetros / pesos del modelo |
| Rango objetivo [5–7 rebotes] | Función de pérdida (loss function) |
| Muestreo por rechazo | Optimización por búsqueda aleatoria |
| Integración Monte Carlo | Estimación de expectativas sobre parámetros |
| Media μ de parámetros aceptados | Estimación robusta del parámetro óptimo |
| Varianza σ² de parámetros aceptados | Incertidumbre / indicador de overfitting |
| Convergencia de la varianza | Convergencia del entrenamiento |
| `dt` en integración de Euler | Tasa de aprendizaje α en Gradiente Descendente |

---

*Material de apoyo para el Módulo 1 — Introducción al Machine Learning a través del Modelado Computacional.*
