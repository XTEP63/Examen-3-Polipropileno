# Predicción del precio del Polipropileno con Deep Learning (LSTM y CNN 1D)

Este proyecto entrena dos modelos para predecir el precio diario del **polipropileno (PP)** a partir de su histórico:

- Una red **LSTM** (Long Short-Term Memory).
- Una red **CNN 1D** (red convolucional 1D).

El objetivo es modelar el precio y generar predicciones a corto plazo (5 días), incluyendo una tabla con los precios pronosticados para las fechas del **24 al 28 de noviembre de 2025**.

---

## 1. Datos y contexto

- Fuente: dataset de Alphacast filtrado a la industria **Industrial == "Polypropylene"**.
- Rango usado en el modelo: a partir de **START_DATE = 2021-01-11**, para trabajar con un histórico reciente y con menos cambios.

---

## 2. Preprocesamiento

### 2.1 Corte temporal y orden

1. Se convierten las fechas a datetime y se ordena el DataFrame por Date.
2. Se aplica el corte desde START_DATE para quedarse con un tramo más homogéneo de mercado.

### 2.2 División Train / Val / Test

Para respetar la naturaleza de serie de tiempo **no se barajan los datos**.  
Se divide la serie de precios en tres bloques contiguos:

- TRAIN_FRACTION = 0.7
- VAL_FRACTION   = 0.15
- TEST_FRACTION  = 0.15

Esto evita fuga de información (no se entrena con datos “del futuro”).

### 2.3 Escalado (StandardScaler)

Se usa **StandardScaler** (media 0, desviación estándar 1) solo con el bloque de entrenamiento:

```python
scaler = StandardScaler()
scaler.fit(prices_train_raw)
```

Luego se transforma train, val y test con ese scaler ya ajustado:

- Ventaja: estabiliza la magnitud de los precios y facilita el entrenamiento de redes profundas.
- Se evita data leakage porque el escalador nunca “ve” datos de validación ni test al ajustarse.

### 2.4 Ventanas deslizantes (supervisado)

La serie escalada sigue siendo 1D (un precio por día).
Para convertirla en problema supervisado se usa una ventana deslizante:

```python
X[i] = [p_{t-W+1}, ..., p_t]
y[i] = p_{t+1}
```

Hay dos flujos:

- Primero se separa en train / val / test SIN escalar ni ventanear.
- vLuego se aplica escalado dentro de cada bloque y se crean las ventanas sobre los precios escalados.

Finalmente se le da la forma requerida por Keras:

```python
X_*_seq.shape = (n_samples, WINDOW_SIZE, 1)  # 1 feature
```

## 3. Elección de WINDOW_SIZE

En este proyecto se usan dos tamaños de ventana distintos, uno por arquitectura:

- LSTM: WINDOW_SIZE = 5
- CNN 1D: WINDOW_SIZE = 25

### 3.1 Por qué 5 pasos para LSTM

- La LSTM tiene celdas con memoria interna y puertas (input/forget/output) que le permiten mantener información relevante más allá de la ventana explícita.

- Con 5 días se captura suficiente contexto de corto plazo (tendencia reciente, micro-momentum) sin reducir demasiado el número de muestras disponibles.

- Ventanas muy largas en una serie relativamente ruidosa pueden meter mucho “ruido” histórico que no aporta al pronóstico inmediato y complican el entrenamiento.

En resumen, la LSTM compensa la ventana corta con su capacidad de memoria interna.

### 3.2 Por qué 25 pasos para CNN 1D

- La CNN 1D no tiene memoria explícita; aprende patrones locales a través de filtros (kernels).

- Con WINDOW_SIZE = 25 se le da a los filtros suficiente “campo de visión” para detectar patrones dentro de aproximadamente 1 mes de datos (25 días).

- Los kernels de tamaño 5 y 3 pueden combinarse: una primera capa detecta patrones en intervalos de 5 días; la segunda combina a una escala.

## 4. Modelo LSTM
### 4.1 Arquitectura
```python
input_shape = (WINDOW_SIZE, 1)  # (5, 1)

lstm_model = Sequential()
lstm_model.add(Input(shape=input_shape))

lstm_model.add(LSTM(units=64, return_sequences=True))
lstm_model.add(LSTM(units=32, return_sequences=False))

lstm_model.add(Dense(32, activation="relu"))
lstm_model.add(Dense(1))  # precio del día siguiente
```

Razonamiento por capas:

- Input(shape=(5, 1))
  - 5 timesteps, 1 feature (precio). Es la secuencia de los últimos 5 días.

- LSTM(64, return_sequences=True)
  - Primera capa recurrente que procesa toda la ventana y genera una secuencia de 5 estados ocultos.
  - 64 unidades es un tamaño intermedio: suficiente capacidad para capturar patrones pero sin caer en sobre-parametrización para el tamaño de dataset.

- LSTM(32, return_sequences=False)

  - Segunda capa que recibe la secuencia previa y la comprime en un solo vector (estado final).

  - Reducir a 32 unidades fuerza cierta compresión de la información, actuando como “resumen” de la dinámica de esos 5 días.

- Dense(32, activation="relu")

  - Capa totalmente conectada que aprende una transformación no lineal sobre el embedding temporal para ajustar detalles y relaciones no lineales.

- Dense(1)

  - Capa de salida lineal que produce el precio normalizado del día siguiente.

En el código hay capas Dropout(0.15) comentadas. Se mantienen como opción para regularizar si en alguna ejecución futura se detecta overfitting, pero en el experimento final no se usan para no perder capacidad de ajuste en un dataset relativamente pequeño.

### 4.2 Hiperparámetros de entrenamiento (LSTM)

```python 
lstm_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse"
)

history_lstm = lstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=200,
    batch_size=32,
    shuffle=False,
    verbose=1
)
```

- Optimizador Adam

  - Maneja bien escalas distintas y ruido en gradientes, típico en series financieras/commodities.

- Learning rate = 0.0005

  - Lo suficientemente pequeño para que el entrenamiento sea estable, pero no tan pequeño como para necesitar miles de épocas.

- Loss: MSE

  - Penaliza fuerte los errores grandes, lo que es útil cuando queremos evitar predicciones muy alejadas del valor real.

- Batch size = 32

  - Tamaño estándar que balancea estabilidad en los gradientes y eficiencia computacional.

- shuffle=False

  - Crítico para series de tiempo: mantener el orden temporal de las secuencias.

## 5. Modelo CNN 1D
### 5.1 Arquitectura

```python
input_shape = (WINDOW_SIZE, 1)  # (25, 1)

cnn_model = Sequential()
cnn_model.add(Input(shape=input_shape))

cnn_model.add(Conv1D(filters=128, kernel_size=5,
                     activation="relu", padding="same"))
cnn_model.add(Dropout(0.1))

cnn_model.add(Conv1D(filters=64, kernel_size=3,
                     activation="relu", padding="same"))
cnn_model.add(Dropout(0.1))

cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation="relu"))
cnn_model.add(Dense(1))
```

Razonamiento por capas:

- Ventana de 25 días

  - Da suficiente contexto para que las convoluciones vean patrones semanales y mensuales.

- Conv1D(128, kernel_size=5, padding="same")

  - 128 filtros permiten capturar muchos tipos de patrones locales (subidas/bajadas abruptas, spikes, micro-tendencias).

  - kernel_size=5 mira bloques de 5 días, similar a una “semana” de trading, lo que tiene sentido económico.

- Dropout(0.1)

  - Apaga aleatoriamente el 10% de las activaciones, ayudando a evitar overfitting en un modelo con bastantes filtros.

- Conv1D(64, kernel_size=3, padding="same")

  - Segunda capa que combina y refina patrones más pequeños (3 días), algo así como “filtrar el ruido” dentro de la ventana grande.

  - Se reduce el número de filtros (128 → 64) porque la representación ya es más rica.

- Otro Dropout(0.1)

  - De nuevo, regularización en la parte convolucional.

- Flatten()

  - Pasa de mapas de características 1D a un vector.

- Dense(128, relu)

  - Capa densa grande que mezcla toda la información de los filtros para producir una representación global de la ventana.

- Dense(1)

  - Salida lineal con el precio normalizado del día siguiente.

### 5.2 Hiperparámetros de entrenamiento (CNN 1D)
```python
cnn_model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss="mse"
)

es = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history_cnn = cnn_model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=400,
    batch_size=32,
    shuffle=False,
    callbacks=[es],
    verbose=1
)
```
- Mismo optimizador y LR que LSTM

  - Facilita comparar comportamientos de ambas arquitecturas bajo condiciones similares.

- Epochs máximas = 400 + EarlyStopping

  - Se permite entrenar más tiempo, pero EarlyStopping detiene cuando la val_loss deja de mejorar durante 20 épocas, evitando sobreentrenamiento y ahorro de tiempo.

- restore_best_weights=True

  - El modelo final es el mejor que se obtuvo en validación, no el de la última época.

## 6. Tabla de predicciones 24–28 Nov 2025

A partir de los modelos entrenados se generó una tabla con las predicciones de ambos modelos para las fechas objetivo:

| Fecha futura | (LSTM)Predicción del precio ($/ton) | (CNN)Predicción del precio ($/ton) | 
|-------------:|-------------------------------------:|------------------------------------:|
| 2025-11-24   | 6383.90                             | 6376.48                              | 
| 2025-11-25   | 6372.10                             | 6358.17                              | 
| 2025-11-26   | 6359.63                             | 6341.96                              | 
| 2025-11-27   | 6347.92                             | 6327.69                              | 
| 2025-11-28   | 6336.62                             | 6315.31                              | 






