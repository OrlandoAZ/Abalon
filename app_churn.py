"""
============================================================
  app_churn.py  —  Despliegue del modelo MLP (Regresión)
  Modelo  : modelo_MLP_churn.keras
  Pickle  : preprocesamiento_churn.pkl
  Target  : Rings (número de anillos del abulón → edad)
============================================================

Ejecutar en terminal:
    streamlit run app_churn.py
"""

import os

import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# ── Constantes de rutas ──────────────────────────────────────────────────────
MODEL_PATH  = "modelo_regresion.keras"
PICKLE_PATH = "preprocesamiento_regresion.pkl"

# ── Rangos de referencia del dataset Abalone (UCI) ───────────────────────────
RANGES = {
    "Length":         (0.075, 0.815),
    "Diameter":       (0.055, 0.650),
    "Height":         (0.000, 1.130),
    "Whole_weight":   (0.002, 2.826),
    "Shucked_weight": (0.001, 1.488),
    "Viscera_weight": (0.001, 0.760),
    "Shell_weight":   (0.002, 1.005),
}

# ── Carga única (cacheada) de modelo y preprocesador ─────────────────────────
@st.cache_resource(show_spinner="Cargando modelo...")
def load_assets():
    """
    Estrategia de carga con tres fallbacks para manejar diferencias entre
    Keras 2 (guardado) y Keras 3 (entorno de ejecución):
      1. tf.keras.models.load_model  con compile=False  [preferido]
      2. tf_keras.models.load_model  con compile=False  [si hay keras 3 standalone]
      3. Lanza error descriptivo con instrucción de solución
    """
    model = None
    try:
        # Intento 1: tf.keras con compile=False (evita error en métricas de Keras 3)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e1:
        try:
            # Intento 2: tf_keras (paquete de compatibilidad de Keras)
            import tf_keras  # pip install tf_keras
            model = tf_keras.models.load_model(MODEL_PATH, compile=False)
        except ImportError:
            raise RuntimeError(
                f"No se pudo cargar el modelo.\n\n"
                f"Error original: {e1}\n\n"
                "Soluciones posibles:\n"
                "  A) En la terminal del entorno virtual:\n"
                "       pip install tf_keras\n"
                "  B) O fija la versión de Keras compatible:\n"
                "       pip install keras==2.15.0\n"
                "  C) O re-entrena el modelo con la versión actual de Keras."
            )
        except Exception as e2:
            raise RuntimeError(
                f"Falló el intento 1 ({e1}) y el intento 2 ({e2}).\n"
                "Verifica que el archivo .keras no esté corrupto."
            )

    with open(PICKLE_PATH, "rb") as f:
        pre = pickle.load(f)
    return model, pre


# ── Función de preprocesamiento (replica exacta del entrenamiento) ────────────
def preprocess(sex: str, inputs: dict, pre: dict) -> np.ndarray:
    """
    Recibe:
        sex    : 'F', 'M' o 'I'
        inputs : dict con las 7 variables numéricas
        pre    : dict del pickle (scaler, num_cols, dummy_cols, feature_columns)

    Devuelve:
        X_scaled : np.ndarray de shape (1, n_features), dtype float32
    """
    scaler          = pre["scaler"]
    num_cols        = pre["num_cols"]
    dummy_cols      = pre["dummy_cols"]   # p. ej. ['Sex_I', 'Sex_M']
    feature_columns = pre["feature_columns"]  # num_cols + dummy_cols

    # 1. Construir vector numérico en el orden de num_cols
    num_values = np.array([[inputs[c] for c in num_cols]], dtype=float)

    # 2. Escalar solo columnas numéricas
    num_scaled = scaler.transform(num_values)

    # 3. Construir dummies (drop_first=True → 'Sex_F' fue la categoría base)
    #    dummy_cols puede ser ['Sex_I', 'Sex_M'] según el orden de pd.get_dummies
    dummy_values = np.zeros((1, len(dummy_cols)), dtype=np.float32)
    for i, col in enumerate(dummy_cols):
        suffix = col.split("_", 1)[1]   # 'I' o 'M'
        if sex == suffix:
            dummy_values[0, i] = 1.0

    # 4. Concatenar y convertir a float32
    X = np.hstack([num_scaled, dummy_values]).astype(np.float32)
    return X


# ════════════════════════════════════════════════════════════════════════════
#  INTERFAZ STREAMLIT
# ════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Predicción de Edad — Abulón",
        page_icon="🐚",
        layout="centered",
    )

    # ── Encabezado ────────────────────────────────────────────────────────────
    st.title("🐚 Predicción de Edad del Abulón")
    st.markdown(
        "Este modelo MLP estima el **número de anillos** del abulón "
        "a partir de sus medidas físicas. "
        "La edad en años se obtiene como **Anillos + 1.5**."
        "Elaborado por Orlando Advíncula Zeballos"
    )
    st.divider()

    # ── Carga de assets ───────────────────────────────────────────────────────
    try:
        model, pre = load_assets()
    except (FileNotFoundError, OSError) as e:
        st.error(
            f"❌ No se encontró el archivo: **{getattr(e, 'filename', str(e))}**\n\n"
            "Asegúrate de que `modelo_MLP_churn.keras` y "
            "`preprocesamiento_churn.pkl` están en la misma carpeta que este script."
        )
        st.stop()
    except RuntimeError as e:
        st.error(f"❌ Error al cargar el modelo:\n\n```\n{e}\n```")
        st.stop()

    # ── Formulario de entrada ─────────────────────────────────────────────────
    st.subheader("📋 Ingresa las medidas del abulón")

    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox(
            "Sexo",
            options=["F", "M", "I"],
            format_func=lambda x: {"F": "F — Hembra", "M": "M — Macho", "I": "I — Infante"}[x],
            help="F = Hembra, M = Macho, I = Infante (juvenil)"
        )

        length = st.number_input(
            "Length (mm)",
            min_value=0.0, max_value=1.0, value=0.52, step=0.001, format="%.3f",
            help=f"Rango típico: {RANGES['Length'][0]} – {RANGES['Length'][1]}"
        )
        diameter = st.number_input(
            "Diameter (mm)",
            min_value=0.0, max_value=1.0, value=0.41, step=0.001, format="%.3f",
            help=f"Rango típico: {RANGES['Diameter'][0]} – {RANGES['Diameter'][1]}"
        )
        height = st.number_input(
            "Height (mm)",
            min_value=0.0, max_value=1.5, value=0.14, step=0.001, format="%.3f",
            help=f"Rango típico: {RANGES['Height'][0]} – {RANGES['Height'][1]}"
        )
        whole_weight = st.number_input(
            "Whole Weight (g)",
            min_value=0.0, max_value=3.0, value=0.80, step=0.001, format="%.4f",
            help=f"Rango típico: {RANGES['Whole_weight'][0]} – {RANGES['Whole_weight'][1]}"
        )

    with col2:
        shucked_weight = st.number_input(
            "Shucked Weight (g)",
            min_value=0.0, max_value=2.0, value=0.35, step=0.001, format="%.4f",
            help=f"Rango típico: {RANGES['Shucked_weight'][0]} – {RANGES['Shucked_weight'][1]}"
        )
        viscera_weight = st.number_input(
            "Viscera Weight (g)",
            min_value=0.0, max_value=1.0, value=0.18, step=0.001, format="%.4f",
            help=f"Rango típico: {RANGES['Viscera_weight'][0]} – {RANGES['Viscera_weight'][1]}"
        )
        shell_weight = st.number_input(
            "Shell Weight (g)",
            min_value=0.0, max_value=1.5, value=0.23, step=0.001, format="%.4f",
            help=f"Rango típico: {RANGES['Shell_weight'][0]} – {RANGES['Shell_weight'][1]}"
        )

    # ── Botón de predicción ───────────────────────────────────────────────────
    st.divider()
    predict_btn = st.button("🔮 Predecir Edad", use_container_width=True, type="primary")

    if predict_btn:
        inputs = {
            "Length":         length,
            "Diameter":       diameter,
            "Height":         height,
            "Whole_weight":   whole_weight,
            "Shucked_weight": shucked_weight,
            "Viscera_weight": viscera_weight,
            "Shell_weight":   shell_weight,
        }

        with st.spinner("Calculando predicción..."):
            X_scaled = preprocess(sex, inputs, pre)
            rings_raw  = float(model.predict(X_scaled, verbose=0).flatten()[0])
            rings_pred = int(np.round(rings_raw))   # redondear primero
            age_pred   = rings_pred + 1.5           # sumar 1.5 al entero

        # ── Resultado principal ───────────────────────────────────────────────
        st.success("✅ Predicción completada")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="🔢 Anillos predichos",
                value=f"{rings_pred:.2f}",
                help="Salida directa de la red neuronal"
            )
        with res_col2:
            st.metric(
                label="📅 Edad estimada (años)",
                value=f"{age_pred:.2f}",
                help="Anillos + 1.5  (fórmula estándar del dataset Abalone)"
            )

        # ── Interpretación ────────────────────────────────────────────────────
        st.info(
            f"El modelo estima **{rings_pred:.1f} anillos**, "
            f"lo que equivale a aproximadamente **{age_pred:.1f} años** de edad."
        )

    # ── Pie de página ─────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Modelo: MLP (Red Neuronal Multicapa) · "
        "Dataset: Abalone UCI · "
        "Arquitectura: Dense(64→32→16→1, ReLU + Dropout)"
    )


if __name__ == "__main__":
    main()

# pip install streamlit
# pip install tf_keras

# En Consola
#   python fix_modelo.py
#        ✅ Modelo cargado exitosamente. Ahora usa 'modelo_regresion_fixed.keras' en la app.
#  modelo original (.keras Keras 2)
#    + Keras 3 instalada          → error batch_shape
#    + TF_USE_LEGACY_KERAS=1      → error "Keras cannot be imported"
#    ──────────────────────────────────────────────────────
#    modelo_fixed (.keras Keras 3)
#    + Keras 3 instalada          → ✅ carga correcta
#    (sin el env var)

#pip freeze > requirements.txt
# Reemplazar requeriments.txt por: pip show streamlit tensorflow scikit-learn numpy pandas
# streamlit==1.45.1
# tensorflow== 1.58.0
# scikit-learn==1.9.0
# numpy==2.4.6
# pandas==3.0.3




#   streamlit run app_churn.py
