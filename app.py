import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import hashlib
import os
from supabase import create_client, Client
import pandas as pd
import matplotlib.pyplot as plt

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
if 'mostrar_feedback' not in st.session_state:
    st.session_state['mostrar_feedback'] = False
if 'ultimo_nombre' not in st.session_state:
    st.session_state['ultimo_nombre'] = None
if 'ultima_fecha' not in st.session_state:
    st.session_state['ultima_fecha'] = None
if 'ultimo_id_prediccion' not in st.session_state:
    st.session_state['ultimo_id_prediccion'] = None

# Credenciales de administrador
ADMIN_PASSWORD = "12345678"
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

# Configurar Supabase
@st.cache_resource
def init_supabase() -> Client:
    """Inicializa la conexión a Supabase"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        st.error("⚠️ Error: Credenciales de Supabase no configuradas")
        st.stop()
    
    return create_client(url, key)

supabase = init_supabase()

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

# Funciones de Supabase
def guardar_prediccion(nombre, probabilidad, fecha_hora):
    """Guarda una predicción en Supabase"""
    try:
        resultado = "Parkinson detectado" if probabilidad > 0.5 else "Saludable"
        
        data = {
            "nombre": nombre,
            "probabilidad": float(probabilidad),
            "resultado": resultado,
            "fecha_hora": fecha_hora,
            "feedback": None  # Inicialmente sin feedback
        }
        
        response = supabase.table("predicciones").insert(data).execute()
        # Guardar el ID de la predicción para asociar feedback después
        if response.data and len(response.data) > 0:
            st.session_state['ultimo_id_prediccion'] = response.data[0]['id']
        return True
    except Exception as e:
        st.error(f"Error al guardar: {str(e)}")
        return False

def guardar_feedback(prediccion_id, feedback_texto):
    """Actualiza el feedback de una predicción en Supabase"""
    try:
        response = supabase.table("predicciones").update(
            {"feedback": feedback_texto}
        ).eq("id", prediccion_id).execute()
        return True
    except Exception as e:
        st.error(f"Error al guardar feedback: {str(e)}")
        return False

def obtener_historial():
    """Obtiene el historial de predicciones desde Supabase"""
    try:
        response = supabase.table("predicciones").select("*").order("fecha_hora", desc=False).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error al cargar historial: {str(e)}")
        return []

def limpiar_historial():
    """Limpia todo el historial en Supabase"""
    try:
        # Primero obtener todos los IDs
        response = supabase.table("predicciones").select("id").execute()
        if response.data:
            for item in response.data:
                supabase.table("predicciones").delete().eq("id", item["id"]).execute()
        return True
    except Exception as e:
        st.error(f"Error al limpiar historial: {str(e)}")
        return False

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

# Opciones de navegación
pagina = st.sidebar.radio("Ir a:", ["🔍 Análisis", "📊 Historial / Panel Admin"])

# ==================== PÁGINA DE ANÁLISIS ====================
if pagina == "🔍 Análisis":
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
            
            if st.button("🔍 Predecir", type="primary"):
                if not nombre_paciente:
                    st.warning("⚠️ Por favor ingresa el nombre del paciente antes de predecir.")
                else:
                    with st.spinner("Analizando imagen..."):
                        probabilidad = predecir_imagen(imagen)
                        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Guardar en Supabase
                        if guardar_prediccion(nombre_paciente, probabilidad, fecha_hora):
                            # Guardar datos para feedback
                            st.session_state['ultimo_nombre'] = nombre_paciente
                            st.session_state['ultima_fecha'] = fecha_hora
                            st.session_state['mostrar_feedback'] = True
                            
                            st.markdown("---")
                            if probabilidad > 0.5:
                                st.error(f"🧠 **Probabilidad de Parkinson detectada: {probabilidad*100:.2f}%**")
                                st.info(f"📅 Análisis realizado el {fecha_hora}")
                            else:
                                st.success(f"✅ **Imagen saludable detectada: {(1 - probabilidad)*100:.2f}%**")
                                st.info(f"📅 Análisis realizado el {fecha_hora}")
                            
                            st.success("💾 Predicción guardada en la base de datos")
                            st.markdown("---")
                            st.markdown("**Nota:** Este resultado es orientativo y no sustituye una evaluación médica profesional.", unsafe_allow_html=True)
        
        # === 🗣️ BLOQUE DE FEEDBACK DEL USUARIO ===
        if st.session_state.get("mostrar_feedback", False):
            st.markdown("---")
            st.markdown("### 🗣️ Retroalimentación del usuario")
            st.info("💡 Tu opinión nos ayuda a mejorar el sistema")
            
            col_fb1, col_fb2 = st.columns([1, 3])
            with col_fb1:
                feedback_correcto = st.radio(
                    "¿Fue correcta la predicción?",
                    ["👍 Sí", "👎 No"],
                    horizontal=True
                )
            with col_fb2:
                comentario = st.text_input(
                    "Comentario opcional:",
                    placeholder="Ej: El resultado fue preciso / Se requiere más análisis...",
                    key="comentario_feedback"
                )
            
            if st.button("📩 Enviar Feedback", type="primary"):
                feedback_texto = f"{feedback_correcto} | {comentario if comentario else 'Sin comentario'}"
                if st.session_state['ultimo_id_prediccion']:
                    if guardar_feedback(st.session_state['ultimo_id_prediccion'], feedback_texto):
                        st.success("✅ ¡Gracias por tu retroalimentación!")
                        st.balloons()
                        st.session_state["mostrar_feedback"] = False
                        st.rerun()
    
    with col2:
        st.markdown("### 📋 Instrucciones")
        st.info("""
        1. Ingresa el nombre del paciente
        2. Sube una imagen clara del trazo
        3. Haz clic en 'Predecir'
        4. Revisa el resultado
        5. Proporciona feedback (opcional)
        6. Ve al historial para ver todos los análisis
        """)
        
        st.markdown("### ℹ️ Sobre la detección")
        st.markdown("""
        - **Espirales:** Patrones de dibujo en espiral
        - **Ondas:** Trazos ondulados
        - Los resultados se guardan automáticamente en la nube
        - Tu feedback mejora el sistema
        """)

# ==================== PÁGINA DE HISTORIAL CON LOGIN ====================
elif pagina == "📊 Historial / Panel Admin":
    
    # Si NO está autenticado, mostrar formulario de login
    if not st.session_state['autenticado']:
        st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔐 Acceso al Historial</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ingresa la contraseña para acceder al historial y panel de administración</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔑 Autenticación requerida")
            
            password_input = st.text_input("Contraseña:", type="password", placeholder="Ingresa la contraseña de administrador")
            
            if st.button("🔓 Acceder", type="primary"):
                if password_input:
                    if iniciar_sesion(password_input):
                        st.success("✅ ¡Acceso concedido!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
                else:
                    st.warning("⚠️ Por favor ingresa una contraseña")
            
            st.markdown("---")
            st.info("💡 **Nota:** Solo el personal autorizado puede acceder al historial de pacientes.")
    
    # Si está autenticado, mostrar el contenido con tabs
    else:
        st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>👤 Panel de Administración</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Bienvenido, {st.session_state['usuario_actual']}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Crear tabs para organizar el contenido
        tab1, tab2, tab3 = st.tabs(["📊 Historial de Predicciones", "📈 Estadísticas y Drift", "💬 Feedback de Usuarios"])
        
        # ==================== TAB 1: HISTORIAL ====================
        with tab1:
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
                        if st.session_state.get('confirmar_limpieza_hist'):
                            if limpiar_historial():
                                st.session_state['confirmar_limpieza_hist'] = False
                                st.success("✅ Historial limpiado")
                                st.rerun()
                        else:
                            st.session_state['confirmar_limpieza_hist'] = True
                            st.warning("⚠️ Clic nuevamente para confirmar")
                
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
        
        # ==================== TAB 2: ESTADÍSTICAS Y DRIFT ====================
        with tab2:
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
                
                # === 🔁 GRÁFICO DE EVOLUCIÓN TEMPORAL (DRIFT) ===
                st.markdown("---")
                st.markdown("### 📈 Evolución temporal de predicciones (Drift del modelo)")
                
                historial = obtener_historial()
                if len(historial) < 2:
                    st.info("📊 Aún no hay suficientes datos para graficar la evolución.")
                else:
                    df = pd.DataFrame(historial)
                    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
                    df = df.sort_values("fecha_hora")
                    df["promedio_movil"] = df["probabilidad"].rolling(window=3, min_periods=1).mean()

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(df["fecha_hora"], df["probabilidad"], color="gray", alpha=0.6, s=50, label="Predicciones individuales")
                    ax.plot(df["fecha_hora"], df["promedio_movil"], marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6, label="Promedio móvil (ventana=3)")
                    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Umbral 50%')
                    ax.set_title("Evolución de las predicciones en el tiempo", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Fecha y hora del análisis", fontsize=11)
                    ax.set_ylabel("Probabilidad de Parkinson", fontsize=11)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

                    primeros = df["probabilidad"].head(3).mean()
                    ultimos = df["probabilidad"].tail(3).mean()
                    drift_valor = (ultimos - primeros) * 100

                    if abs(drift_valor) < 5:
                        estado = "✅ Sin drift significativo"
                        color = "green"
                    elif drift_valor > 5:
                        estado = "⚠️ Posible aumento en detecciones de Parkinson"
                        color = "orange"
                    else:
                        estado = "⚠️ Posible disminución en detecciones de Parkinson"
                        color = "orange"

                    st.markdown(f"<p style='color:{color};font-weight:bold;font-size:16px;'>📊 {estado} ({drift_valor:+.2f}% de variación promedio)</p>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Últimos análisis
                st.markdown("### 🕐 Últimos 5 Análisis")
                historial = obtener_historial()
                ultimos = historial[-5:][::-1] if len(historial) >= 5 else historial[::-1]
                
                for pred in ultimos:
                    color = "#ff4b4b" if pred['probabilidad'] > 0.5 else "#00cc00"
                    st.markdown(f"""
                    <div style='padding: 10px; border-left: 4px solid {color}; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 8px;'>
                        <strong>{pred['nombre']}</strong> - {pred['fecha_hora']} - <strong>{pred['probabilidad']*100:.2f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        # ==================== TAB 3: FEEDBACK DE USUARIOS ====================
        with tab3:
            st.markdown("### 💬 Retroalimentación de Usuarios")
            
            historial = obtener_historial()
            feedback_data = [p for p in historial if p.get('feedback') and p['feedback'].strip()]
            
            if len(feedback_data) == 0:
                st.info("📭 No se ha recibido feedback todavía.")
            else:
                # Estadísticas de feedback
                positivos = sum(1 for f in feedback_data if "👍" in f['feedback'])
                negativos = sum(1 for f in feedback_data if "👎" in f['feedback'])
                total_fb = len(feedback_data)
                
                st.markdown("### 📊 Resumen de Feedback")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📝 Total feedback", total_fb)
                with col2:
                    st.metric("✅ Predicciones correctas", f"{positivos} ({positivos/total_fb*100:.1f}%)")
                with col3:
                    st.metric("❌ Predicciones incorrectas", f"{negativos} ({negativos/total_fb*100:.1f}%)")
                
                st.markdown("---")
                st.markdown("### 💭 Comentarios recibidos")
                
                for fb in reversed(feedback_data):
                    icono = "👍" if "👍" in fb["feedback"] else "👎"
                    color = "#28a745" if "👍" in fb["feedback"] else "#dc3545"
                    
                    # Extraer comentario
                    partes = fb["feedback"].split("|")
                    comentario = par
