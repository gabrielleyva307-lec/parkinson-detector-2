import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Detector de Parkinson AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para hacer la interfaz más bonita
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .result-positive {
        background-color: #fee;
        color: #c53030;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #fc8181;
        margin: 1rem 0;
    }
    
    .result-negative {
        background-color: #f0fff4;
        color: #2f855a;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #68d391;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo (con cache para que sea más rápido)
@st.cache_resource
def load_parkinson_model():
    try:
        model = load_model('modelo_parkinson.h5')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función de preprocesamiento
def preprocess(img):
    img = img.resize((224, 224)).convert('RGB')
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Función para hacer predicción
def predict_parkinson(image, model):
    try:
        img_array = preprocess(image)
        prediction = model.predict(img_array)[0][0]
        return prediction
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None

# Interfaz principal
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Detector de Parkinson AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Análisis inteligente de imágenes médicas usando Deep Learning</p>', unsafe_allow_html=True)
    
    # Cargar modelo
    model = load_parkinson_model()
    
    if model is None:
        st.error("❌ No se pudo cargar el modelo. Verifica que el archivo 'modelo_parkinson.h5' esté en el directorio.")
        st.stop()
    
    st.success("✅ Modelo cargado correctamente")
    
    # Divisor
    st.markdown("---")
    
    # Upload de imagen
    st.subheader("📁 Sube tu imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen para analizar",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        
        # Crear dos columnas para mejor layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Imagen Original")
            st.image(image, caption="Imagen subida", use_column_width=True)
        
        with col2:
            st.subheader("📊 Información")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tamaño:** {image.size}")
            st.write(f"**Formato:** {image.format}")
            st.write(f"**Modo:** {image.mode}")
        
        # Botón de análisis
        st.markdown("---")
        
        if st.button("🔍 Analizar Imagen", key="analyze_btn"):
            with st.spinner("🤖 Analizando imagen... Por favor espera"):
                prediction = predict_parkinson(image, model)
                
                if prediction is not None:
                    # Mostrar resultado
                    st.markdown("### 📋 Resultado del Análisis")
                    
                    if prediction > 0.5:
                        # Parkinson detectado
                        st.markdown(f"""
                        <div class="result-positive">
                            <h3>⚠️ Parkinson Detectado</h3>
                            <p><strong>Confianza:</strong> {prediction:.2%}</p>
                            <p><em>Se recomienda consultar con un especialista médico para confirmación.</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Saludable
                        st.markdown(f"""
                        <div class="result-negative">
                            <h3>✅ Resultado Normal</h3>
                            <p><strong>Confianza:</strong> {(1-prediction):.2%}</p>
                            <p><em>No se detectaron indicadores de Parkinson en la imagen.</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Barra de progreso visual
                    st.markdown("#### 📈 Nivel de Confianza")
                    confidence = prediction if prediction > 0.5 else (1 - prediction)
                    st.progress(confidence)
                    
                    # Información adicional
                    with st.expander("ℹ️ Información sobre el análisis"):
                        st.write("""
                        **¿Cómo funciona?**
                        - Este modelo usa redes neuronales convolucionales (CNN)
                        - Analiza patrones en la imagen que pueden indicar Parkinson
                        - La confianza indica qué tan seguro está el modelo
                        
                        **Importante:**
                        - Este es un análisis preliminar, NO un diagnóstico médico
                        - Siempre consulta con un profesional de la salud
                        - Los resultados pueden variar según la calidad de la imagen
                        """)

    else:
        # Mostrar información mientras no hay imagen
        st.info("👆 Sube una imagen para comenzar el análisis")
        
        with st.expander("📖 ¿Cómo usar esta herramienta?"):
            st.write("""
            1. **Sube una imagen** usando el botón de arriba
            2. **Revisa la vista previa** para asegurarte de que se cargó correctamente
            3. **Haz clic en "Analizar"** para obtener el resultado
            4. **Interpreta los resultados** con la ayuda de un profesional médico
            
            **Tipos de imagen recomendados:**
            - Imágenes claras y bien iluminadas
            - Formato JPG, PNG o JPEG
            - Tamaño apropiado (no muy pequeñas)
            """)

# Sidebar con información adicional
def sidebar():
    st.sidebar.markdown("### 🧠 Detector de Parkinson")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("""
    **Versión:** 1.0  
    **Modelo:** Deep Learning CNN  
    **Precisión:** ~85%  
    
    ---
    
    **⚠️ Disclaimer:**  
    Esta herramienta es solo para fines educativos y de investigación. No sustituye el diagnóstico médico profesional.
    """)
    
    if st.sidebar.button("🔄 Limpiar Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache limpiado!")

# Ejecutar la aplicación
if __name__ == "__main__":
    sidebar()
    main()