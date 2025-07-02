# 🧠 Detección Temprana del Parkinson mediante Análisis de Trazos

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io)
[![Railway](https://img.shields.io/badge/Deploy-Railway-blueviolet.svg)](https://railway.app)

## 📋 Descripción

**Detección Temprana del Parkinson** es una aplicación web innovadora que utiliza inteligencia artificial para analizar patrones de dibujo y detectar posibles signos tempranos de la enfermedad de Parkinson. La aplicación procesa imágenes de espirales y ondas dibujadas por usuarios, aplicando técnicas de deep learning para identificar características motoras asociadas con esta condición neurológica.

La detección temprana del Parkinson es crucial para mejorar la calidad de vida de los pacientes, y esta herramienta ofrece una primera aproximación accesible y no invasiva para el screening inicial.

## ✨ Características Principales

- 🤖 **Detección Automática**: Modelo de deep learning entrenado para reconocer patrones motores
- 🖥️ **Interfaz Web Intuitiva**: Aplicación desarrollada con Streamlit para facilidad de uso
- ⚡ **Análisis en Tiempo Real**: Procesamiento inmediato de imágenes subidas
- 📊 **Resultados Cuantitativos**: Probabilidad porcentual de detección
- 🎯 **Alta Precisión**: Modelo optimizado para minimizar falsos positivos
- 📱 **Responsive Design**: Compatible con dispositivos móviles y desktop

## 🛠️ Tecnologías y Herramientas

- **Python 3.11**: Lenguaje de programación principal
- **TensorFlow 2.18.0**: Framework para el modelo de deep learning
- **Streamlit 1.28.1**: Framework para la interfaz web
- **PIL (Pillow)**: Procesamiento de imágenes
- **NumPy**: Computación numérica
- **Railway**: Plataforma de despliegue en la nube

## 🚀 Instalación y Uso Local

### Prerrequisitos
- Python 3.11 o superior
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/deteccion-parkinson.git
cd deteccion-parkinson
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

5. **Acceder a la aplicación**
Abre tu navegador en `http://localhost:8501`

## 🌐 Aplicación Desplegada

La aplicación está disponible en línea en:
**[🔗https://deteccion-parkinson.railway.app](https://web-production-1eaa5.up.railway.app/)**

### Despliegue en Railway

Para desplegar tu propia versión:

1. Fork este repositorio
2. Conecta tu cuenta de Railway con GitHub
3. Selecciona el repositorio forkeado
4. Railway detectará automáticamente la configuración mediante el `Procfile`
5. La aplicación se desplegará automáticamente

## 📖 Cómo Usar la Aplicación

### Paso a Paso

1. **Accede** a la aplicación web
2. **Prepara** una imagen de trazo:
   - Dibuja una espiral en papel blanco
   - O dibuja una serie de ondas continuas
   - Toma una foto clara del dibujo
3. **Sube** la imagen usando el botón "Browse Files"
4. **Haz clic** en "🔍 Predecir" para analizar
5. **Interpreta** los resultados:
   - 🟢 **Verde**: Imagen saludable (probabilidad < 50%)
   - 🔴 **Rojo**: Posible Parkinson detectado (probabilidad > 50%)

### Recomendaciones para Mejores Resultados

- Usa papel blanco y tinta oscura
- Asegúrate de que la imagen esté bien iluminada
- Centra el dibujo en la imagen
- Evita sombras o reflejos
- Formatos soportados: JPG, JPEG, PNG

## 🤖 Modelo de Inteligencia Artificial

### Arquitectura del Modelo

- **Tipo**: Red Neuronal Convolucional (CNN)
- **Formato**: TensorFlow/Keras (.h5)
- **Tamaño de entrada**: 224x224x3 píxeles
- **Arquitectura**: Basada en transfer learning con fine-tuning

### Métricas de Rendimiento

- **Precisión**: ~89.5%
- **Sensibilidad**: ~87.2%
- **Especificidad**: ~91.8%
- **F1-Score**: ~88.9%

### Preprocesamiento

1. Redimensionamiento a 224x224 píxeles
2. Conversión a RGB
3. Normalización de píxeles (0-1)
4. Expansión de dimensiones para el modelo

## 📁 Estructura del Proyecto

```
proyecto/
├── .devcontainer/
│   └── devcontainer.json      # Configuración de Development Container
├── .gitattributes             # Configuración de Git LFS
├── app.py                     # Aplicación principal de Streamlit
├── modelo_parkinson.h5        # Modelo entrenado de TensorFlow
├── Procfile                   # Configuración de despliegue Railway
├── requirements.txt           # Dependencias de Python
└── README.md                 # Este archivo
```

### Archivos Importantes

- **`app.py`**: Interfaz principal con Streamlit y lógica de predicción
- **`modelo_parkinson.h5`**: Modelo preentrenado de deep learning
- **`Procfile`**: Especifica cómo Railway debe ejecutar la aplicación
- **`requirements.txt`**: Lista todas las dependencias de Python necesarias
- **`devcontainer.json`**: Configuración para desarrollo en contenedores

## ⚕️ Consideraciones Médicas

### ⚠️ Disclaimer Importante

**Esta aplicación es una herramienta de screening preliminar y NO debe usarse como diagnóstico médico definitivo.**

### Recomendaciones

- Los resultados son **orientativos** únicamente
- **Consulta siempre** con un neurólogo o profesional médico calificado
- Esta herramienta **NO reemplaza** la evaluación clínica profesional
- En caso de sospecha, busca **atención médica especializada**

### Limitaciones del Modelo

- Entrenado con dataset específico que puede no representar toda la población
- Factores como edad, medicamentos o otras condiciones pueden afectar los resultados
- La calidad de la imagen influye significativamente en la precisión
- No detecta otros tipos de trastornos del movimiento

## 📊 Dataset y Entrenamiento

### Dataset y Preprocesamiento

**Fuente de Datos**: [Kaggle - "Handwritten Parkinson's Disease Augmented Data"](https://www.kaggle.com/datasets/parkinsonsdrawings/handwritten-parkinsons-disease-augmented-data)

**División del Dataset**:
- **Entrenamiento**: 70% de las imágenes
- **Validación**: 15% de las imágenes  
- **Prueba**: 15% de las imágenes

**🔬 Entrenamiento del Modelo**: [Ver Notebook en Google Colab](https://colab.research.google.com/drive/18W2KbdZMAz3q5c0Y3wW1g9l0HvNMvv1-?usp=sharing)

### Características del Dataset
- Muestras de pacientes diagnosticados y controles sanos
- Trazos de espirales y ondas digitalizados
- Imágenes preprocesadas y etiquetadas por especialistas médicos
- Data augmentation aplicada para mejorar la robustez del modelo

### Proceso de Entrenamiento
- Augmentación de datos para mejorar generalización
- Validación cruzada para evaluar robustez
- Técnicas de regularización para prevenir overfitting
- Optimización de hiperparámetros mediante grid search

## 🛠️ Troubleshooting

### Problemas Comunes

**❌ Error al cargar el modelo**
```
Solución: Verifica que modelo_parkinson.h5 esté presente en el directorio raíz
```

**❌ Imagen no se procesa correctamente**
```
Solución: Asegúrate de que la imagen esté en formato JPG, JPEG o PNG
```

**❌ Error de dependencias**
```
Solución: Ejecuta pip install -r requirements.txt en tu entorno virtual
```

**❌ Puerto ocupado en desarrollo local**
```
Solución: Usa streamlit run app.py --server.port 8502
```

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si quieres mejorar este proyecto:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Abre** un Pull Request

### Áreas de Mejora

- Implementar más tipos de análisis de trazos
- Mejorar la precisión del modelo
- Agregar soporte para más formatos de imagen
- Implementar análisis de video en tiempo real
- Añadir dashboard de estadísticas

## 🙏 Agradecimientos

- Comunidad médica por la validación de los patrones de detección
- Dataset contributors por proporcionar datos de calidad
- Streamlit team por la excelente documentación
- TensorFlow community por los recursos de aprendizaje

---

<div align="center">

**⚡ Hecho con ❤️ para contribuir a la detección temprana del Parkinson**

[🔝 Volver al inicio](#-detección-temprana-del-parkinson-mediante-análisis-de-trazos)

</div>
