# Rama `etl_JP`: Procesamiento ETL y Clasificador k-NN Desde Cero

**Autor:** José Pablo  
**Curso:** Inteligencia Artificial Avanzada para Ciencia de Datos  
**Reto:** Clasificación de Movimientos de Rehabilitación Post-ACV (REHAB)  
**Rama:** `etl_JP`  

---

## 1. Resumen de la Tarea

En esta rama se desarrolla un flujo completo e independiente para:
1. **Extracción y Estructuración (ETL)**: Transformar las señales temporales multicanal $(N \times 880 \times 12)$ del dataset cinemático REHAB en un nuevo dataset tabular $(N \times 72)$ optimizado para algoritmos clásicos de Machine Learning.
2. **Implementación de Machine Learning desde Cero**: Programar un clasificador **$k$-Nearest Neighbors ($k$-NN)** multiclase, un estandarizador $Z$-score y módulos de evaluación (Accuracy, Precision, Recall, F1-Score y Matriz de Confusión) en Python puro y operaciones numéricas básicas con NumPy, **sin utilizar ninguna librería de Machine Learning** (como `scikit-learn`, `scipy.spatial` o `statsmodels`).
3. **Ejecución Autónoma por Consola**: Scripts `.py` ejecutables directamente desde la terminal mediante cualquier intérprete estándar de Python, sin requerir Jupyter Notebooks ni IDEs específicos.

---

## 2. Estructura y Justificación del Nuevo Dataset Tabular

### 2.1. Motivación de la Reducción Dimensional
El dataset crudo consiste en series temporales donde cada muestra representa una repetición de ejercicio capturada a lo largo de 880 puntos temporales a través de 12 canales sensoriales (6 de sensores inerciales IMU y 6 del guante de flexión).

Para aplicar modelos clásicos de aprendizaje máquina, cada serie temporal se sintetiza en un vector de **características biomecánicas** que capturan la postura estática, el rango articular y la energía dinámica del movimiento.

### 2.2. Sensores y Canales (12 canales)
| Índice | Canal | Sensor / Dispositivo | Descripción Biomecánica |
|---|---|---|---|
| `ch00` | `imu1_pitch` | IMU 1 (Brazo / Muslo) | Ángulo de elevación / flexión del segmento proximal |
| `ch01` | `imu1_yaw` | IMU 1 (Brazo / Muslo) | Rotación horizontal del segmento proximal |
| `ch02` | `imu1_roll` | IMU 1 (Brazo / Muslo) | Rotación axial / pronación del segmento proximal |
| `ch03` | `imu2_pitch` | IMU 2 (Antebrazo / Pantorrilla) | Ángulo de elevación del segmento distal |
| `ch04` | `imu2_yaw` | IMU 2 (Antebrazo / Pantorrilla) | Rotación horizontal del segmento distal |
| `ch05` | `imu2_roll` | IMU 2 (Antebrazo / Pantorrilla) | Rotación axial del segmento distal |
| `ch06` | `glove_f1_thumb` | Guante Sensorial | Grado de flexión del Dedo 1 (Pulgar) |
| `ch07` | `glove_f2_index` | Guante Sensorial | Grado de flexión del Dedo 2 (Índice) |
| `ch08` | `glove_f3_middle` | Guante Sensorial | Grado de flexión del Dedo 3 (Medio) |
| `ch09` | `glove_f4_ring` | Guante Sensorial | Grado de flexión del Dedo 4 (Anular) |
| `ch10` | `glove_f5_pinky` | Guante Sensorial | Grado de flexión del Dedo 5 (Meñique) |
| `ch11` | `glove_s5_pitch` | Guante Sensorial (S5) | Inclinación vertical (pitch) de la palma/mano |

### 2.3. Estadísticas Extraídas por Canal (6 estadísticas)
Para cada canal $c \in \{0, \dots, 11\}$ sobre los $T = 880$ puntos de tiempo de la repetición:

1. **Media (`mean`)**:
   $$\mu_c = \frac{1}{T} \sum_{t=1}^T x_t$$
   *Significado:* Posición promedio u orientación de referencia del miembro o articulación durante la repetición.
2. **Desviación Estándar (`std`)**:
   $$\sigma_c = \sqrt{\frac{1}{T} \sum_{t=1}^T (x_t - \mu_c)^2}$$
   *Significado:* Medida del dinamismo, dispersión y grado de fluctuación de la trayectoria.
3. **Mínimo (`min`)**:
   $$\min_c = \min_{t} (x_t)$$
   *Significado:* Límite inferior alcanzado por la articulación o sensor.
4. **Máximo (`max`)**:
   $$\max_c = \max_{t} (x_t)$$
   *Significado:* Límite superior de extensión/flexión alcanzado.
5. **Rango de Movimiento (`range`)**:
   $$ROM_c = \max_c - \min_c$$
   *Significado:* Amplitud total recorrida por la articulación en el ejercicio.
6. **Valor Cuadrático Medio / Energía (`rms`)**:
   $$RMS_c = \sqrt{\frac{1}{T} \sum_{t=1}^T x_t^2}$$
   *Significado:* Potencia o magnitud global de la señal cinemática.

### 2.4. Estructura Final del Archivo CSV
- **Ubicación:** `data/rehab_tabular_features.csv`
- **Dimensiones:** $4,257$ registros (filas) $\times 75$ columnas.
- **Columnas:**
  - `sample_id`: Identificador entero único de la muestra ($0$ a $4,256$).
  - `ch00_imu1_pitch_mean` hasta `ch11_glove_s5_pitch_rms`: 72 variables continuas ($12 \text{ canales} \times 6 \text{ estadísticas}$).
  - `movement_id`: Etiqueta numérica objetivo ($0$ a $15$, excluyendo el 14 por dato ausente en la fuente original).
  - `movement_name`: Nombre textual descriptivo de la acción clínica realizada.

---

## 3. Implementación del Algoritmo $k$-NN Desde Cero

El clasificador fue programado íntegramente en [`src/knn_from_scratch.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/REHAB-RETO/src/knn_from_scratch.py) cubriendo los siguientes bloques algorítmicos:

### 3.1. Estandarización $Z$-Score (`StandardScalerManual`)
Evita que variables con magnitudes angulares mayores dominen el cálculo de distancias:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j + \epsilon}$$
*Regla de integridad:* $\mu_j$ y $\sigma_j$ se calculan exclusivamente en el conjunto de entrenamiento (80%) y se utilizan para transformar tanto entrenamiento como prueba, evitando contaminación de datos (*data leakage*).

### 3.2. División Estratificada Train / Test (`train_test_split_manual`)
Divide los datos (80% entrenamiento = 3,406 muestras, 20% prueba = 851 muestras) manteniendo la proporción exacta de cada una de las 15 clases de movimientos mediante particionado balanceado reproducible (`random_state=42`).

### 3.3. Clasificador $k$-NN (`KNNClassifierManual`)
- **Cálculo de Distancia Euclidiana ($L_2$):**
  $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^D (u_j - v_j)^2}$$
- **Votación Ponderada por Inversa de la Distancia:**
  $$w_i = \frac{1}{d_i + \epsilon}$$
  La clase predicha $\hat{y}$ corresponde a:
  $$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_k(x), y_i = c} w_i$$
- **Probabilidad Estimada por Clase:**
  $$P(y = c \mid x) = \frac{\sum_{i \in \mathcal{N}_k(x), y_i = c} w_i}{\sum_{j \in \mathcal{N}_k(x)} w_j}$$

### 3.4. Métricas de Evaluación Manuales
Cálculo directo sin dependencias externas:
- **Exactitud (Accuracy):** $\frac{\sum_{i=1}^M \mathbb{I}(y_i = \hat{y}_i)}{M}$
- **Matriz de Confusión $C$:** Matriz $15 \times 15$ donde $C_{ij}$ denota la cantidad de muestras de la clase real $i$ clasificadas como $j$.
- **Precision, Recall y F1-Score:**
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

---

## 4. Resultados de la Evaluación

Al evaluar el clasificador con $k = 5$, métrica Euclidiana y ponderación por distancia sobre las **851 muestras del conjunto de prueba independiente**:

### 4.1. Métricas Globales
- **Exactitud Global (Accuracy):** **95.18%** (810 aciertos de 851 muestras).
- **Macro Average F1-Score:** **94.99%**
- **Weighted Average F1-Score:** **95.21%**

### 4.2. Desempeño por Tipo de Movimiento
| ID | Movimiento de Rehabilitación | Precision | Recall | F1-Score | Soporte (Test) |
|---|---|---|---|---|---|
| `00` | Bobath Handshake | 1.0000 | 1.0000 | **1.0000** | 46 |
| `01` | Bobath Flexion/Extension | 0.8810 | 0.8810 | **0.8810** | 42 |
| `02` | Bobath Forward Flexion/Extension | 0.9123 | 0.9811 | **0.9455** | 53 |
| `03` | Bobath Anterior/Posterior Rotation | 1.0000 | 0.9600 | **0.9796** | 50 |
| `04` | Elbow Flexion and Wrist Compression | 0.9455 | 0.9123 | **0.9286** | 57 |
| `05` | Wrist Flexion and Extension | 0.9649 | 0.9322 | **0.9483** | 59 |
| `06` | Finger-to-Finger Training | 0.9259 | 0.9615 | **0.9434** | 52 |
| `07` | Ball Gripping | 0.9625 | 1.0000 | **0.9809** | 77 |
| `08` | Shoulder Joint Internal/External Rotation | 0.9655 | 0.9333 | **0.9492** | 60 |
| `09` | Breast Expansion | 0.9839 | 1.0000 | **0.9919** | 61 |
| `10` | Flexion-Pressure Rotation Forward/Back | 1.0000 | 0.9677 | **0.9836** | 62 |
| `11` | Elbow Joint Flexion and Touch | 0.8103 | 1.0000 | **0.8952** | 47 |
| `12` | Shoulder Touch Training | 0.9444 | 0.8644 | **0.9027** | 59 |
| `13` | Ankle Extension & Knee Int/Ext Rotation | 1.0000 | 0.9524 | **0.9756** | 63 |
| `15` | Hip Flexion and Extension | 0.9667 | 0.9206 | **0.9431** | 63 |

---

## 5. Instrucciones de Ejecución

El código puede compilarse/ejecutarse directamente en la terminal sin depender de ningún entorno gráfico ni notebook.

### 5.1. Ejecución Rápida de Todo el Pipeline
Para generar el dataset tabular, entrenar el modelo desde cero y visualizar predicciones paso a paso:
```bash
python run_pipeline.py
```

### 5.2. Opciones de Personalización por Consola
Puedes ajustar hiperparámetros directamente desde la línea de comandos:
```bash
# Probar con k=7, metrica Manhattan y 8 muestras de demostracion
python run_pipeline.py -k 7 --metric manhattan --demo-samples 8

# Forzar la re-extraccion de caracteristicas desde los archivos .npy
python run_pipeline.py --force-etl
```

### 5.3. Ejecución Directa de los Módulos Individuales
```bash
# 1. Solo extraccion ETL:
python src/etl_features.py data data/rehab_tabular_features.csv

# 2. Solo clasificador k-NN:
python src/knn_from_scratch.py --csv data/rehab_tabular_features.csv -k 5 --metric euclidean
```

---

## 6. Estructura de Archivos en la Rama `etl_JP`

```
REHAB-RETO/
├── README.md                      # Documentacion completa de la rama etl_JP
├── requirements.txt               # Requisitos minimos de ejecucion
├── run_pipeline.py                # Punto de entrada principal para ejecucion integral
├── data/
│   ├── README.md                  # Referencia al origen de datos
│   ├── *.npy                      # Archivos de senales crudas descargados
│   └── rehab_tabular_features.csv # Dataset procesado generado (4257 x 75)
└── src/
    ├── data_loader.py             # Funciones de lectura de senales crudas
    ├── etl_features.py            # Pipeline ETL de extraccion estadistica
    └── knn_from_scratch.py        # Algoritmo k-NN, normalizador, metricas y CLI
```