# Modelo de Probabilidad de Default (PD) y Scorecard Crediticio

## Resumen del Proyecto

Implementación end-to-end de un modelo de riesgo crediticio para estimar la Probabilidad de Default (PD) de solicitantes de préstamos. El sistema incluye el procesamiento de datos brutos, la transformación de variables mediante Weight of Evidence (WoE), el entrenamiento de un modelo de Regresión Logística interpretable y el despliegue de una interfaz interactiva construida con Streamlit.

Diseñado para su uso en entornos financieros, permite la evaluación visual del rendimiento del modelo y la simulación en tiempo real de puntuaciones crediticias (Credit Scores).

## Conjunto de Datos (Lending Club)

El proyecto utiliza datos de Lending Club, una plataforma de préstamos peer-to-peer en EE. UU. El conjunto de datos histórico (2007-2014) contiene 466,285 observaciones y 74 variables, detallando el estado de los préstamos (Vigente, Atrasado, Pagado, etc.), información de pagos y características sociodemográficas y financieras de los solicitantes.

Variables clave utilizadas en el modelado:
* Características del préstamo: `loanAmnt` (Monto), `term` (Plazo), `intRate` (Tasa de interés), `grade` (Calificación LC), `purpose` (Propósito).

* Perfil del solicitante: `empLength` (Antigüedad laboral), `homeOwnership` (Situación habitacional), `annualInc` (Ingreso anual), `addrState` (Estado de residencia).

* Historial crediticio: `dti` (Deuda a ingresos), `delinq2Yrs` (Atrasos en 2 años), `inqLast6Mths` (Consultas en 6 meses), `openAcc` (Líneas de crédito abiertas), `pubRec` (Registros públicos despectivos).
## Arquitectura y Tecnologías

* **Lenguaje:** Python 3
* **Modelado:** Scikit-Learn, SciPy
* **Manipulación de Datos:** Pandas, NumPy
* **Visualización:** Plotly, Matplotlib, Seaborn
* **Despliegue de Interfaz:** Streamlit

## Características Principales (App Streamlit)

1. **Análisis de Variables (WoE):** Visualización interactiva del Peso de la Evidencia (Weight of Evidence) para variables categóricas y continuas discretizadas, permitiendo evaluar la relación lineal de cada atributo con la variable objetivo.
2. **Métricas de Desempeño:** Cálculo y graficación en tiempo real de métricas estándar de la industria de riesgo:
   * Área bajo la curva ROC (AUROC).
   * Coeficiente de Gini.
   * Estadística de Kolmogorov-Smirnov (KS).
   * Estrategia de Cutoffs basada en tasas de aprobación y rechazo.
3. **Scorecard Operativa:** Tabla de puntuación derivada de los coeficientes del modelo de regresión logística, escalada a un rango estándar 300 - 850 puntos.
4. **Simulador de Clientes:** Motor de inferencia que permite el ingreso manual de características de un solicitante para calcular instantáneamente su Probabilidad de Default y su Credit Score final.

## Metodología

1. **Preprocesamiento:** Limpieza de datos, imputación de valores nulos y creación de variables indicadoras (dummies) agrupadadas categóricamente.
2. **Ingeniería de Características:** Cálculo de WoE e Information Value (IV) para maximizar la separación entre las clases ("Good" vs "Bad" loans).
3. **Modelado:** Regresión Logística estándar aumentada con el cálculo de valores-p (p-values) mediante la matriz de información de Fisher para validación estadística de los coeficientes.
4. **Escalamiento del Score:** Transformación lineal de las probabilidades logarítmicas (log-odds) generadas por el modelo en un sistema de puntuación discreto, utilizando un puntaje base y un factor de escalamiento (PDO - Points to Double the Odds).

## Estructura del Repositorio

    ├── app.py                  # Script principal de la aplicación Streamlit
    ├── model.py                # Lógica del modelo (Regresión Logística, métricas, scorecard)
    ├── utils.py                # Funciones auxiliares (Carga de datos, cálculo WoE, preprocesamiento)
    ├── plots.py                # Funciones de visualización con Plotly
    ├── pd_model.sav            # Modelo entrenado serializado
    ├── data/                   # Directorio con los sets de entrenamiento y prueba
    │   ├── data.rar
    |   ├── LCDataDictionary.xlsx
    └── .streamlit/
        └── config.toml         # Configuración del tema oscuro de la interfaz
## Imagenes ilustrativas
![image alt](https://github.com/johnnyhuincahue/Probability-Default-Model/blob/68fc6ff3da5f2d8a4cfd9d3fae652ed2dd7d0f61/images/tab2.PNG)
![image alt](https://github.com/johnnyhuincahue/Probability-Default-Model/blob/18c9774021266146e0cf0bb0f0c1582174308030/images/tab1.PNG)
![image alt](https://github.com/johnnyhuincahue/Probability-Default-Model/blob/18c9774021266146e0cf0bb0f0c1582174308030/images/tab3.PNG)
![image alt](https://github.com/johnnyhuincahue/Probability-Default-Model/blob/18c9774021266146e0cf0bb0f0c1582174308030/images/tab4.PNG)
## Instalación y Ejecución

1. Clonar el repositorio.

2. Descomprimir el archivo rar que contiene los datos.

3. Instalar las dependencias requeridas:
    pip install pandas numpy scikit-learn scipy plotly streamlit

4. Ejecutar la aplicación:
    streamlit run app.py
