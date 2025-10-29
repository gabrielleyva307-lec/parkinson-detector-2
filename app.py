import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import json
import hashlib

# Configuración de la página
st.set_page_config(
    page_title="Detector de Parkinson",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state para autenticación
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False
if 'usuario_actual' not in st.session_state:
    st.session_state['usuario_actual'] = None

# Credenciales de administrador (contraseña hasheada)
ADMIN_PASSWORD = "12345678"
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_parkinson.h5")
    return modelo

modelo = cargar_modelo()

# Funciones de autenticación
def verificar_contraseña(password):
    """Verifica si la contraseña es correcta"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == ADMIN_PASSWORD_HASH

def iniciar_sesion(password):
    """Inicia sesión del administrador"""
    if verificar_contraseña(password):
        st.session_state['autenticado'] = True
        st.session_state['usuario_actual'] = "Administrador"
        return True
    return False

def cerrar_sesion():
    """Cierra la sesión del administrador"""
    st.session_state['autenticado'] = False
    st.session_state['usuario_actual'] = None

# Funciones de almacenamiento
def guardar_prediccion(nombre, probabilidad, fecha_hora):
    """Guarda una predicción en el historial"""
    try:
        resultado = st.session_state.get('historial_storage')
        if resultado is None:
            historial = []
        else:
            historial = json.loads(resultado)
    except:
        historial = []
    
    nueva_prediccion = {
        "nombre": nombre,
        "probabilidad": float(probabilidad),
        "fecha_hora": fecha_hora,
        "resultado": "Parkinson detectado" if probabilidad > 0.5 else "Saludable"
    }
    historial.append(nueva_prediccion)
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

def obtener_estadisticas_avanzadas():
    """Obtiene estadísticas detalladas del sistema"""
    historial = obtener_historial()
    
    if len(historial) == 0:
        return None
    
    total_analisis = len(historial)
    total_parkinson = sum(1 for p in historial if p['probabilidad'] > 0.5)
    total_saludable = total_analisis - total_parkinson
    prob_promedio = sum(p['probabilidad'] for p in historial) / total_analisis
    prob_maxima = max(p['probabilidad'] for p in historial)
    prob_minima = min(p['probabilidad'] for p in historial)
    
    # Análisis por rangos
    rangos = {
        "0-25%": 0,
        "25-50%": 0,
        "50-75%": 0,
        "75-100%": 0
    }
    
    for p in historial:
        prob = p['probabilidad'] * 100
        if prob < 25:
            rangos["0-25%"] += 1
        elif prob < 50:
            rangos["25-50%"] += 1
        elif prob < 75:
            rangos["50-75%"] += 1
        else:
            rangos["75-100%"] += 1
    
    # Pacientes únicos
    pacientes_unicos = len(set(p['nombre'] for p in historial))
    
    return {
        "total_analisis": total_analisis,
        "total_parkinson": total_parkinson,
        "total_saludable": total_saludable,
        "prob_promedio": prob_promedio,
        "prob_maxima": prob_maxima,
        "prob_minima": prob_minima,
        "rangos": rangos,
        "pacientes_unicos": pacientes_unicos
    }

# Sidebar para navegación
st.sidebar.title("🧠 Navegación")

# Mostrar estado de sesión en sidebar
if st.session_state['autenticado']:
    st.sidebar.success(f"👤 {st.session_state['usuario_actual']}")
    if st.sidebar.button("🚪 Cerrar Sesión"):
        cerrar_sesion()
        st.rerun()

# Opciones de navegación según el estado de autenticación
if st.session_state['autenticado']:
    pagina = st.sidebar.radio("Ir a:", ["🔍 Análisis", "📊 Historial", "👤 Panel Admin"])
else:
    pagina = st.sidebar.radio("Ir a:", ["🔍 Análisis", "📊 Historial", "🔐 Iniciar Sesión"])

# ==================== PÁGINA DE LOGIN ====================
if pagina == "🔐 Iniciar Sesión":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔐 Acceso Administrador</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔑 Credenciales de acceso")
        
        password_input = st.text_input("Contraseña:", type="password", placeholder="Ingresa la contraseña")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔓 Iniciar Sesión", type="primary", use_column_width=True):
                if password_input:
                    if iniciar_sesion(password_input):
                        st.success("✅ ¡Acceso concedido!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
                else:
                    st.warning("⚠️ Por favor ingresa una contraseña")
        
        with col_btn2:
            if st.button("🔙 Volver", use_column_width=True):
                st.rerun()
        
        st.markdown("---")
        st.info("💡 **Nota:** Esta sección es solo para administradores del sistema.")

# ==================== PÁGINA DE ANÁLISIS ====================
elif pagina == "🔍 Análisis":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Detección de Parkinson</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Sube una imagen de trazo para predecir la probabilidad de Parkinson.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        nombre_paciente = st.text_input("👤 Nombre del paciente:", placeholder="Ej: Juan Pérez")
        imagen_subida = st.file_uploader("📤 Sube una imagen (trazo de espiral u onda)", type=["jpg", "jpeg", "png"])
        
        if imagen_subida:
            imagen = Image.open(imagen_subida)
            st.image(imagen, caption='Imagen cargada', use_column_width=True)
            
            if st.button("🔍 Predecir", type="primary", use_column_width=True):
                if not nombre_paciente:
                    st.warning("⚠️ Por favor ingresa el nombre del paciente antes de predecir.")
                else:
                    with st.spinner("Analizando imagen..."):
                        probabilidad = predecir_imagen(imagen)
                        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        guardar_prediccion(nombre_paciente, probabilidad, fecha_hora)
                        
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
        
        col_filtro, col_boton = st.columns([3, 1])
        
        with col_filtro:
            filtro = st.selectbox("🔍 Filtrar por resultado:", 
                                 ["Todos", "Parkinson detectado", "Saludable"])
        
        with col_boton:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Limpiar historial", type="secondary"):
                limpiar_historial()
                st.rerun()
        
        historial_invertido = historial[::-1]
        
        st.markdown("### 📋 Registro de análisis")
        
        for idx, prediccion in enumerate(historial_invertido):
            if filtro == "Parkinson detectado" and prediccion['probabilidad'] <= 0.5:
                continue
            elif filtro == "Saludable" and prediccion['probabilidad'] > 0.5:
                continue
            
            if prediccion['probabilidad'] > 0.5:
                color_borde = "#ff4b4b"
                icono = "🔴"
            else:
                color_borde = "#00cc00"
                icono = "🟢"
            
            with st.container():
                st.markdown(f"""
                <div style='padding: 15px; border-left: 5px solid {color_borde}; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;'>
                    <p style='margin: 0; font-size: 18px;'><strong>{icono} {prediccion['nombre']}</strong></p>
                    <p style='margin: 5px 0; color: #666;'>📅 {prediccion['fecha_hora']}</p>
                    <p style='margin: 5px 0;'><strong>Resultado:</strong> {prediccion['resultado']}</p>
                    <p style='margin: 5px 0;'><strong>Probabilidad de Parkinson:</strong> {prediccion['probabilidad']*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💾 Exportar datos")
        
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

# ==================== PANEL ADMIN ====================
elif pagina == "👤 Panel Admin":
    if not st.session_state['autenticado']:
        st.warning("⚠️ Debes iniciar sesión para acceder al panel de administración.")
        st.info("👉 Ve a la sección '🔐 Iniciar Sesión' en el menú lateral.")
    else:
        st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>👤 Panel de Administración</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Bienvenido, {st.session_state['usuario_actual']}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        stats = obtener_estadisticas_avanzadas()
        
        if stats is None:
            st.info("📊 No hay datos disponibles aún. Espera a que se realicen análisis.")
        else:
            # Métricas principales
            st.markdown("### 📊 Métricas Generales")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Pacientes únicos", stats['pacientes_unicos'])
            with col2:
                st.metric("📈 Total análisis", stats['total_analisis'])
            with col3:
                st.metric("🔴 Casos positivos", stats['total_parkinson'])
            with col4:
                tasa_deteccion = (stats['total_parkinson'] / stats['total_analisis']) * 100
                st.metric("📊 Tasa detección", f"{tasa_deteccion:.1f}%")
            
            st.markdown("---")
            
            # Estadísticas de probabilidades
            st.markdown("### 🎯 Estadísticas de Probabilidad")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Promedio", f"{stats['prob_promedio']*100:.2f}%")
            with col2:
                st.metric("⬆️ Máxima", f"{stats['prob_maxima']*100:.2f}%")
            with col3:
                st.metric("⬇️ Mínima", f"{stats['prob_minima']*100:.2f}%")
            
            st.markdown("---")
            
            # Distribución por rangos
            st.markdown("### 📊 Distribución por Rangos de Probabilidad")
            
            col1, col2, col3, col4 = st.columns(4)
            
            rangos = stats['rangos']
            
            with col1:
                st.markdown(f"""
                <div style='padding: 20px; background-color: #d4edda; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #155724; margin: 0;'>0-25%</h3>
                    <h2 style='color: #155724; margin: 10px 0;'>{rangos['0-25%']}</h2>
                    <p style='color: #155724; margin: 0;'>Muy bajo riesgo</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='padding: 20px; background-color: #d1ecf1; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #0c5460; margin: 0;'>25-50%</h3>
                    <h2 style='color: #0c5460; margin: 10px 0;'>{rangos['25-50%']}</h2>
                    <p style='color: #0c5460; margin: 0;'>Bajo riesgo</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='padding: 20px; background-color: #fff3cd; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #856404; margin: 0;'>50-75%</h3>
                    <h2 style='color: #856404; margin: 10px 0;'>{rangos['50-75%']}</h2>
                    <p style='color: #856404; margin: 0;'>Riesgo moderado</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div style='padding: 20px; background-color: #f8d7da; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #721c24; margin: 0;'>75-100%</h3>
                    <h2 style='color: #721c24; margin: 10px 0;'>{rangos['75-100%']}</h2>
                    <p style='color: #721c24; margin: 0;'>Alto riesgo</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Últimos análisis
            st.markdown("### 🕐 Últimos 5 Análisis")
            historial = obtener_historial()
            ultimos = historial[-5:][::-1]
            
            for pred in ultimos:
                color = "#ff4b4b" if pred['probabilidad'] > 0.5 else "#00cc00"
                st.markdown(f"""
                <div style='padding: 10px; border-left: 4px solid {color}; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 8px;'>
                    <strong>{pred['nombre']}</strong> - {pred['fecha_hora']} - <strong>{pred['probabilidad']*100:.2f}%</strong>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Herramientas de administración
            st.markdown("### 🔧 Herramientas de Administración")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Limpiar todo el historial", type="secondary", use_column_width=True):
                    if st.session_state.get('confirmar_limpieza'):
                        limpiar_historial()
                        st.session_state['confirmar_limpieza'] = False
                        st.success("✅ Historial limpiado correctamente")
                        st.rerun()
                    else:
                        st.session_state['confirmar_limpieza'] = True
                        st.warning("⚠️ Haz clic nuevamente para confirmar")
            
            with col2:
                historial_completo = obtener_historial()
                if historial_completo:
                    historial_json = json.dumps(historial_completo, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Exportar datos (JSON)",
                        data=historial_json,
                        file_name=f"datos_completos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_column_width=True
                    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏥 Información")
st.sidebar.info("Esta aplicación utiliza inteligencia artificial para detectar indicadores de Parkinson en trazos de escritura. Los resultados son orientativos.")
