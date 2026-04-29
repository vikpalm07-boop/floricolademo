import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(page_title="Demo ML Florícola", layout="wide")

st.title("Simulador predictivo para preventa y producción florícola")
st.write("Demo comercial: predice producción exportable, riesgo de quiebre y oportunidades de preventa usando Machine Learning.")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("floricola_dataset_limpio_ml.csv", parse_dates=["fecha","fecha_siembra","fecha_pedido","fecha_entrega"])
    df["mes"] = df["fecha"].dt.month
    df["semana"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["dias_a_entrega"] = (df["fecha_entrega"] - df["fecha_pedido"]).dt.days
    return df

df = cargar_datos()

features = [
    "especie","variedad","tipo_cultivo","edad_dias","area_ha","densidad_plantas_ha",
    "temp_media_c","humedad_relativa_pct","radiacion_mj_m2","lluvia_mm","vpd_kpa",
    "mano_obra_disponible","mano_obra_requerida","disponibilidad_camara_frio_pct",
    "fertilizacion_nivel","aplicacion_fitosanitaria","incidencia_plaga_pct",
    "incidencia_enfermedad_pct","merma_poscosecha_pct","tipo_cliente","pais_destino",
    "canal","tallos_solicitados","precio_unitario","descuento_pct","prioridad_cliente",
    "lead_time_dias","mes","semana","dias_a_entrega"
]

X = df[features]
y_prod = df["produccion_exportable_tallos"]
y_riesgo = df["riesgo_quiebre"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

@st.cache_resource
def entrenar_modelos():
    modelo_produccion = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=3))
    ])
    modelo_riesgo = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=3))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y_prod, test_size=0.2, random_state=42)
    modelo_produccion.fit(X_train, y_train)
    pred = modelo_produccion.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_riesgo, test_size=0.2, random_state=42, stratify=y_riesgo)
    modelo_riesgo.fit(X_train_c, y_train_c)
    acc = accuracy_score(y_test_c, modelo_riesgo.predict(X_test_c))
    return modelo_produccion, modelo_riesgo, mae, r2, acc

modelo_produccion, modelo_riesgo, mae, r2, acc = entrenar_modelos()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros limpios", f"{len(df):,}")
k2.metric("MAE producción", f"{mae:,.0f} tallos")
k3.metric("R² producción", f"{r2:.2f}")
k4.metric("Accuracy riesgo", f"{acc:.2f}")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Escenario comercial y operativo")
    especie = st.selectbox("Especie", sorted(df["especie"].unique()))
    variedad = st.selectbox("Variedad", sorted(df[df["especie"] == especie]["variedad"].unique()))
    tipo_cultivo = st.selectbox("Tipo de cultivo", sorted(df["tipo_cultivo"].unique()))
    tipo_cliente = st.selectbox("Tipo de cliente", sorted(df["tipo_cliente"].unique()))
    pais_destino = st.selectbox("País destino", sorted(df["pais_destino"].unique()))
    canal = st.selectbox("Canal", sorted(df["canal"].unique()))
    prioridad_cliente = st.selectbox("Prioridad cliente", ["Alta", "Media", "Baja"])

    area_ha = st.slider("Área del lote (ha)", 0.2, 2.0, 1.0, 0.1)
    edad_dias = st.slider("Edad del cultivo (días)", 30, 180, 105, 1)
    tallos_solicitados = st.number_input("Tallos solicitados en preventa", 300, 30000, 9000, 100)
    precio_unitario = st.number_input("Precio unitario estimado", 0.05, 1.20, 0.42, 0.01)
    descuento_pct = st.slider("Descuento (%)", 0, 20, 5, 1)
    lead_time_dias = st.slider("Lead time de preventa (días)", 7, 45, 21, 1)

with col2:
    st.subheader("Condiciones de producción")
    temp_media_c = st.slider("Temperatura media °C", 10.0, 24.0, 17.0, 0.1)
    humedad_relativa_pct = st.slider("Humedad relativa %", 45.0, 96.0, 74.0, 0.5)
    radiacion_mj_m2 = st.slider("Radiación MJ/m²", 7.0, 26.0, 17.0, 0.5)
    lluvia_mm = st.slider("Lluvia mm", 0.0, 25.0, 2.0, 0.5)
    vpd_kpa = st.slider("VPD kPa", 0.25, 2.5, 0.8, 0.05)
    mano_obra_disponible = st.slider("Mano de obra disponible", 8, 55, 28, 1)
    mano_obra_requerida = st.slider("Mano de obra requerida", 4, 45, 24, 1)
    disponibilidad_camara_frio_pct = st.slider("Cámara de frío disponible %", 55.0, 100.0, 86.0, 0.5)
    fertilizacion_nivel = st.selectbox("Fertilización", ["Bajo", "Medio", "Alto"], index=1)
    aplicacion_fitosanitaria = st.selectbox("Aplicación fitosanitaria", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    incidencia_plaga_pct = st.slider("Incidencia plaga %", 0.0, 35.0, 8.0, 0.5)
    incidencia_enfermedad_pct = st.slider("Incidencia enfermedad %", 0.0, 30.0, 5.0, 0.5)
    merma_poscosecha_pct = st.slider("Merma poscosecha %", 2.0, 25.0, 7.0, 0.5)

escenario = pd.DataFrame([{
    "especie": especie,
    "variedad": variedad,
    "tipo_cultivo": tipo_cultivo,
    "edad_dias": edad_dias,
    "area_ha": area_ha,
    "densidad_plantas_ha": int(df[df["especie"] == especie]["densidad_plantas_ha"].median()),
    "temp_media_c": temp_media_c,
    "humedad_relativa_pct": humedad_relativa_pct,
    "radiacion_mj_m2": radiacion_mj_m2,
    "lluvia_mm": lluvia_mm,
    "vpd_kpa": vpd_kpa,
    "mano_obra_disponible": mano_obra_disponible,
    "mano_obra_requerida": mano_obra_requerida,
    "disponibilidad_camara_frio_pct": disponibilidad_camara_frio_pct,
    "fertilizacion_nivel": fertilizacion_nivel,
    "aplicacion_fitosanitaria": aplicacion_fitosanitaria,
    "incidencia_plaga_pct": incidencia_plaga_pct,
    "incidencia_enfermedad_pct": incidencia_enfermedad_pct,
    "merma_poscosecha_pct": merma_poscosecha_pct,
    "tipo_cliente": tipo_cliente,
    "pais_destino": pais_destino,
    "canal": canal,
    "tallos_solicitados": tallos_solicitados,
    "precio_unitario": precio_unitario,
    "descuento_pct": descuento_pct,
    "prioridad_cliente": prioridad_cliente,
    "lead_time_dias": lead_time_dias,
    "mes": 2,
    "semana": 6,
    "dias_a_entrega": lead_time_dias + 3
}])

prod_pred = modelo_produccion.predict(escenario)[0]
riesgo_pred = modelo_riesgo.predict(escenario)[0]
gap = prod_pred - tallos_solicitados

st.divider()
r1, r2, r3 = st.columns(3)
r1.metric("Producción exportable predicha", f"{prod_pred:,.0f} tallos")
r2.metric("Gap oferta-demanda", f"{gap:,.0f} tallos")
r3.metric("Riesgo de quiebre", riesgo_pred)

if gap < 0:
    st.error("Recomendación: no confirmar el 100% del pedido. Renegociar volumen/fecha o activar otro lote.")
elif gap > tallos_solicitados * 0.2:
    st.warning("Recomendación: hay riesgo de sobreproducción. Activar preventa adicional o buscar cliente alterno.")
else:
    st.success("Recomendación: pedido viable. Mantener seguimiento climático, sanitario y de mano de obra.")

st.subheader("Datos históricos de referencia")
st.dataframe(df[["fecha","especie","variedad","tipo_cliente","pais_destino","tallos_solicitados","produccion_exportable_tallos","gap_demanda_oferta","riesgo_quiebre","margen_bruto_estimado"]].head(200), use_container_width=True)

st.subheader("Producción promedio por especie")
fig, ax = plt.subplots(figsize=(7, 4))
df.groupby("especie")["produccion_exportable_tallos"].mean().sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Tallos exportables promedio")
st.pyplot(fig)

st.caption("Demo con datos sintéticos para validación comercial. En producción se reemplaza por datos reales de finca, clima, ventas y logística.")
