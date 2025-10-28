import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import json

# Configuración de la página
st.set_page_config(
    page_title="Detector de Parkinson",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_parkinson.h5")
    return modelo

modelo = cargar_modelo()

# Funciones de almacenamiento
def guardar_prediccion(nombre, probabilidad, fecha_hora):
    """Guarda una predicción en el historial"""
    try:
        # Intentar obtener historial existente
        resultado = st.session_state.get('historial_storage')
        if resultado is None:
            historial = []
        else:
            historial = json.loads(resultado)
    except:
        historial = []
    
    # Agregar nueva predicción
    nueva_prediccion = {
        "nombre": nombre,
        "probabilidad": float(probabilidad),
        "fecha_hora": fecha_hora,
        "resultado": "Parkinson detectado" if probabilidad > 0.5 else "Saludable"
    }
    historial.append(nueva_prediccion)
    
    # Guardar en session_state
    st.session_state['historial_storage'] = json.dumps(historial)
    return True

def obtener_historial():
    """Obtiene el historial de predicciones"""
    try:
        resultado = st.session_state.get('historial_storage')
        if resultado is None:
            return []
        return json.loads(resultado)
    except:
        return []

def limpiar_historial():
    """Limpia todo el historial"""
    st.session_state['historial_storage'] = json.dumps([])
    return True

# Función para predicción
def predecir_imagen(imagen):
    img = imagen.convert("RGB").resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = modelo.predict(img_array, verbose=0)[0][0]
    return pred

# Sidebar para navegación
st.sidebar.title("🧠 Navegación")
pagina = st.sidebar.radio("Ir a:", ["🔍 Análisis", "📊 Historial"])

# ==================== PÁGINA DE ANÁLISIS ====================
if pagina == "🔍 Análisis":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Detección de Parkinson</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Sube una imagen de trazo para predecir la probabilidad de Parkinson.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Crear dos columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Campo para el nombre
        nombre_paciente = st.text_input("👤 Nombre del paciente:", placeholder="Ej: Juan Pérez")
        
        # Subir imagen
        imagen_subida = st.file_uploader("📤 Sube una imagen (trazo de espiral u onda)", type=["jpg", "jpeg", "png"])
        
        if imagen_subida:
            imagen = Image.open(imagen_subida)
            st.image(imagen, caption='Imagen cargada', use_container_width=True)
            
            if st.button("🔍 Predecir", type="primary", use_container_width=True):
                if not nombre_paciente:
                    st.warning("⚠️ Por favor ingresa el nombre del paciente antes de predecir.")
                else:
                    with st.spinner("Analizando imagen..."):
                        probabilidad = predecir_imagen(imagen)
                        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Guardar en historial
                        guardar_prediccion(nombre_paciente, probabilidad, fecha_hora)
                        
                        # Mostrar resultado
                        st.markdown("---")
                        if probabilidad > 0.5:
                            st.error(f"🧠 **Probabilidad de Parkinson detectada: {probabilidad*100:.2f}%**")
                            st.info(f"📅 Análisis realizado el {fecha_hora}")
                        else:
                            st.success(f"✅ **Imagen saludable detectada: {(1 - probabilidad)*100:.2f}%**")
                            st.info(f"📅 Análisis realizado el {fecha_hora}")
                        
                        st.success("💾 Predicción guardada en el historial")
                        st.markdown("---")
                        st.markdown("**Nota:** Este resultado es orientativo y no sustituye una evaluación médica profesional.", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Instrucciones")
        st.info("""
        1. Ingresa el nombre del paciente
        2. Sube una imagen clara del trazo
        3. Haz clic en 'Predecir'
        4. Revisa el resultado
        5. Ve al historial para ver todos los análisis
        """)
        
        st.markdown("### ℹ️ Sobre la detección")
        st.markdown("""
        - **Espirales:** Patrones de dibujo en espiral
        - **Ondas:** Trazos ondulados
        - Los resultados se guardan automáticamente
        """)

# ==================== PÁGINA DE HISTORIAL ====================
elif pagina == "📊 Historial":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 Historial de Predicciones</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    historial = obtener_historial()
    
    if len(historial) == 0:
        st.info("📭 No hay predicciones guardadas aún. Realiza tu primer análisis en la pestaña '🔍 Análisis'.")
    else:
        # Estadísticas generales
        col1, col2, col3, col4 = st.columns(4)
        
        total_analisis = len(historial)
        total_parkinson = sum(1 for p in historial if p['probabilidad'] > 0.5)
        total_saludable = total_analisis - total_parkinson
        prob_promedio = sum(p['probabilidad'] for p in historial) / total_analisis
        
        with col1:
            st.metric("📈 Total de análisis", total_analisis)
        with col2:
            st.metric("🔴 Parkinson detectado", total_parkinson)
        with col3:
            st.metric("🟢 Saludables", total_saludable)
        with col4:
            st.metric("📊 Prob. promedio", f"{prob_promedio*100:.1f}%")
        
        st.markdown("---")
        
        # Opciones de filtro
        col_filtro, col_boton = st.columns([3, 1])
        
        with col_filtro:
            filtro = st.selectbox("🔍 Filtrar por resultado:", 
                                 ["Todos", "Parkinson detectado", "Saludable"])
        
        with col_boton:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Limpiar historial", type="secondary"):
                limpiar_historial()
                st.rerun()
        
        # Mostrar historial filtrado (más recientes primero)
        historial_invertido = historial[::-1]
        
        st.markdown("### 📋 Registro de análisis")
        
        for idx, prediccion in enumerate(historial_invertido):
            # Aplicar filtro
            if filtro == "Parkinson detectado" and prediccion['probabilidad'] <= 0.5:
                continue
            elif filtro == "Saludable" and prediccion['probabilidad'] > 0.5:
                continue
            
            # Determinar color según resultado
            if prediccion['probabilidad'] > 0.5:
                color_borde = "#ff4b4b"
                icono = "🔴"
            else:
                color_borde = "#00cc00"
                icono = "🟢"
            
            # Crear tarjeta de predicción
            with st.container():
                st.markdown(f"""
                <div style='padding: 15px; border-left: 5px solid {color_borde}; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;'>
                    <p style='margin: 0; font-size: 18px;'><strong>{icono} {prediccion['nombre']}</strong></p>
                    <p style='margin: 5px 0; color: #666;'>📅 {prediccion['fecha_hora']}</p>
                    <p style='margin: 5px 0;'><strong>Resultado:</strong> {prediccion['resultado']}</p>
                    <p style='margin: 5px 0;'><strong>Probabilidad de Parkinson:</strong> {prediccion['probabilidad']*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Opción de descarga
        st.markdown("---")
        st.markdown("### 💾 Exportar datos")
        
        # Convertir a formato descargable
        historial_texto = "HISTORIAL DE PREDICCIONES - DETECTOR DE PARKINSON\n"
        historial_texto += "=" * 60 + "\n\n"
        
        for pred in historial_invertido:
            historial_texto += f"Paciente: {pred['nombre']}\n"
            historial_texto += f"Fecha: {pred['fecha_hora']}\n"
            historial_texto += f"Resultado: {pred['resultado']}\n"
            historial_texto += f"Probabilidad: {pred['probabilidad']*100:.2f}%\n"
            historial_texto += "-" * 60 + "\n\n"
        
        st.download_button(
            label="📥 Descargar historial (TXT)",
            data=historial_texto,
            file_name=f"historial_parkinson_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏥 Información")
st.sidebar.info("Esta aplicación utiliza inteligencia artificial para detectar indicadores de Parkinson en trazos de escritura. Los resultados son orientativos.")
