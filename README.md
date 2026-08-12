# Reto Parcial 1 - Clasificacion de movimientos de rehabilitacion (REHAB)

**Equipo:** Angel Lugo, Jose Pablo, Juan Pablo
**Curso:** IA Avanzada para Ciencia de Datos
**Metodologia:** CRISP-DM

## Contexto del reto

El reto usa el dataset **REHAB**: señales de sensores portatiles (IMUs + guante de flexion)
recolectadas de 120 pacientes post-ACV durante un programa de rehabilitacion de dos semanas.
Trabajamos unicamente con la parte de **Rehab_exercise** (16 movimientos de entrenamiento de rehabilitacion) y no con la parte de evaluacion clinica FMA de 27 items.

- Paper: Lv et al. (2026), *A wearable sensor-based kinematic dataset collected under
  standardized rehabilitation tasks from 120 post-stroke patients*, Scientific Data 13:1136.
  https://doi.org/10.1038/s41597-026-07802-2
- Dataset: Science Data Bank, https://doi.org/10.57760/sciencedb.37018

**El problema de ML:** dado un tramo de señal de sensores (880 puntos de tiempo x 12
canales), predecir cual de los 16 movimientos de rehabilitacion se esta ejecutando
(clasificacion multiclase).

## Dato corrupto

El archivo `014_1.npy`esta **corrupto en el propio repositorio de Science Data Bank**. Por ahora `data_loader.py` excluye ese movimiento del set principal de 12 canales; ver `src/data_loader.py` para la opcion de incluirlo parcialmente (solo 6 canales del guante). Este hallazgo se documenta como hallazgo de calidad de datos en el reporte del equipo.

## Estructura del repo

```
reto-rehab-clasificacion/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    
│   └── README.md            # instrucciones de donde descargar los datos
├── notebooks/
│   └── 01_data_loading_eda.ipynb   # carga, consolidacion y EDA
├── src/
│   └── data_loader.py       # funciones reutilizables de carga del dataset
└── reports/                 # figuras y reportes generados
```