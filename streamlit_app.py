import os
from datetime import timedelta
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ========================================
# FUNCIONES DE CARGA DE DATOS
# ========================================

@st.cache_data(show_spinner=False)
def load_data_from_google_drive(file_id):
	"""Descarga y carga datos desde Google Drive usando gdown."""
	try:
		with st.spinner('📥 Descargando datos desde Google Drive (~193 MB). Primera vez: 1-3 minutos...'):
			import gdown
			
			# URL directa para gdown
			url = f'https://drive.google.com/uc?id={file_id}'
			
			# Descargar a un objeto en memoria
			output = io.BytesIO()
			gdown.download(url, output, quiet=False, fuzzy=True)
			output.seek(0)
			
			# Leer el CSV comprimido
			df = pd.read_csv(output, compression='gzip')
			st.success("✅ Datos cargados exitosamente desde Google Drive")
			return df
			
	except ImportError:
		st.error("❌ La librería 'gdown' no está instalada. Añade 'gdown' a requirements.txt")
		return None
	except Exception as e:
		st.error(f"❌ Error al cargar datos: {e}")
		st.info("💡 Verifica que el archivo sea accesible públicamente y que el ID sea correcto")
		return None


@st.cache_data(show_spinner=False)
def load_data_from_url_simple(url):
	"""Carga datos desde URL directa (Dropbox, GitHub raw, etc)."""
	try:
		with st.spinner('📥 Descargando datos (~193 MB). Primera vez: 1-3 minutos...'):
			df = pd.read_csv(url, compression='gzip')
			st.success("✅ Datos cargados exitosamente")
			return df
	except Exception as e:
		st.error(f"❌ Error al cargar datos: {e}")
		return None


def process_dataframe(df):
	"""Procesa el dataframe con todas las transformaciones necesarias."""
	if df is None:
		return None
		
	# Mapear columnas a nombres estándar
	column_mapping = {
		'country_region': 'country',
		'province_state': 'province',
		'confirmed': 'confirmed',
		'deaths': 'deaths',
		'recovered': 'recovered',
		'active': 'active'
	}
	
	# Renombrar columnas si existen
	df = df.rename(columns=column_mapping)
	
	# Convertir fecha
	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"])
	else:
		raise ValueError("El dataset debe contener una columna 'date'")
	
	# Asegurar columnas necesarias
	for col in ["confirmed", "deaths"]:
		if col not in df.columns:
			df[col] = 0
		else:
			df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
	
	# Manejar columna recovered
	if "recovered" not in df.columns:
		df["recovered"] = 0
	else:
		df["recovered"] = pd.to_numeric(df["recovered"], errors='coerce').fillna(0)
	
	# Calcular activos si no existe
	if "active" not in df.columns or df["active"].isna().all():
		df["active"] = df["confirmed"] - df["recovered"] - df["deaths"]
	else:
		df["active"] = pd.to_numeric(df["active"], errors='coerce').fillna(0)
	
	# Asegurar que active no sea negativo
	df["active"] = df["active"].clip(lower=0)
	
	# Agregar continente basado en país (mapeo básico)
	continent_map = {
		'Mainland China': 'Asia',
		'China': 'Asia',
		'US': 'America',
		'United States': 'America',
		'Spain': 'Europe',
		'Italy': 'Europe',
		'France': 'Europe',
		'Germany': 'Europe',
		'UK': 'Europe',
		'United Kingdom': 'Europe',
		'Brazil': 'America',
		'India': 'Asia',
		'Russia': 'Europe',
		'South Africa': 'Africa',
		'Mexico': 'America',
		'Peru': 'America',
		'Chile': 'America',
		'Iran': 'Asia',
		'Colombia': 'America',
		'Argentina': 'America',
		'Canada': 'America',
		'Australia': 'Oceania',
		'Japan': 'Asia',
		'South Korea': 'Asia',
		'Thailand': 'Asia',
		'Pakistan': 'Asia',
		'Indonesia': 'Asia',
		'Turkey': 'Europe',
		'Egypt': 'Africa',
		'Nigeria': 'Africa',
		'Philippines': 'Asia',
	}
	
	df['continent'] = df['country'].map(continent_map)
	df['continent'] = df['continent'].fillna('Other')
	
	# Ordenar datos
	df = df.sort_values(["country", "date"]).reset_index(drop=True)
	
	# Agregar datos por país y fecha para simplificar
	df_agg = df.groupby(['country', 'continent', 'date']).agg({
		'confirmed': 'sum',
		'deaths': 'sum',
		'recovered': 'sum',
		'active': 'sum'
	}).reset_index()
	
	# Calcular nuevos casos diarios
	df_agg["new_confirmed"] = df_agg.groupby("country")["confirmed"].diff().fillna(0).clip(lower=0)
	
	return df_agg


# ========================================
# FUNCIONES DE ANÁLISIS
# ========================================

def compute_indicators(df):
	"""Calcula indicadores principales del dataset filtrado."""
	if df.empty:
		return {
			"confirmed": 0,
			"active": 0,
			"recovered": 0,
			"deaths": 0,
			"countries": 0,
		}, pd.DataFrame()
	
	latest = df.groupby(["country"]).last().reset_index()
	total = df["country"].nunique()
	indicators = {
		"confirmed": int(df["confirmed"].sum()),
		"active": int(df["active"].sum()),
		"recovered": int(df["recovered"].sum()),
		"deaths": int(df["deaths"].sum()),
		"countries": total,
	}
	return indicators, latest


def rebrote_indicator(series_new_cases, window_recent=14, window_prev=14, threshold=1.2):
	"""Calcula si hay rebrote comparando medias móviles recientes vs previas."""
	if len(series_new_cases) < (window_recent + window_prev):
		return 0.0, False

	recent_mean = series_new_cases[-window_recent:].mean()
	prev_mean = series_new_cases[-(window_recent + window_prev) : -window_recent].mean()
	if prev_mean == 0:
		return float("inf") if recent_mean > 0 else 0.0, recent_mean > 0
	ratio = recent_mean / prev_mean
	return float(ratio), ratio >= threshold


def growth_rate(series_confirmed, days=7):
	"""Calcula tasa de crecimiento media diaria (%) en los últimos `days` días."""
	s = series_confirmed.dropna()
	if len(s) < days + 1:
		return 0.0
	recent = s[-(days + 1) :]
	daily_pct = recent.pct_change().dropna()
	return float(daily_pct.mean() * 100)


def generate_insights(df_filtered):
	"""Genera insights automáticos basados en los datos filtrados."""
	insights = []
	
	if df_filtered.empty:
		return ["Datos insuficientes para generar análisis."]
	
	# Casos nuevos agregados
	new_cases_total = int(df_filtered["new_confirmed"].sum())
	insights.append(f"Nuevos casos en el rango seleccionado: {new_cases_total:,}")

	# Tendencia global (últimos 14 días)
	agg = df_filtered.groupby("date")["new_confirmed"].sum().sort_index()
	if len(agg) >= 14:
		last14 = agg[-14:].sum()
		prev14 = agg[-28:-14].sum() if len(agg) >= 28 else 0
		if prev14 == 0:
			insights.append("No hay período previo completo para comparar la tendencia de 14 días.")
		else:
			change = (last14 - prev14) / prev14
			pct = change * 100
			sign = "aumento" if change > 0 else "disminución"
			insights.append(f"Comparado con los 14 días previos: {abs(pct):.1f}% de {sign} en nuevos casos.")

	# País con mayor nuevos casos
	by_country = df_filtered.groupby("country")["new_confirmed"].sum()
	if not by_country.empty:
		top = by_country.idxmax()
		insights.append(f"País con más nuevos casos: {top} ({int(by_country.max()):,} casos)")

	return insights


# ========================================
# FUNCIÓN PRINCIPAL
# ========================================

def main():
	st.set_page_config(
		page_title="Panel COVID-19 Global",
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
	# CSS personalizado
	st.markdown("""
		<style>
		/* Fondo principal - gris oscuro */
		.main {
			background-color: #1a1a1a;
		}
		.stApp {
			background-color: #1a1a1a;
		}
		
		/* Títulos */
		h1 {
			color: #ffffff !important;
			font-weight: 300 !important;
			letter-spacing: -0.5px;
			margin-bottom: 0.5rem;
			font-size: 2.5rem !important;
		}
		h3 {
			color: #e0e0e0 !important;
			font-weight: 400;
			margin-top: 2rem;
			margin-bottom: 1rem;
			font-size: 1.3rem;
		}
		
		/* Subtítulo y párrafos */
		[data-testid="stMarkdownContainer"] p {
			color: #b0b0b0 !important;
			font-size: 1rem;
		}
		
		/* Métricas */
		[data-testid="stMetricLabel"] {
			color: #888888 !important;
			font-size: 0.75rem !important;
			text-transform: uppercase;
			letter-spacing: 0.5px;
			font-weight: 500;
		}
		[data-testid="stMetricValue"] {
			font-size: 1.8rem !important;
			color: #ffffff !important;
			font-weight: 600;
		}
		
		/* Sidebar */
		section[data-testid="stSidebar"] {
			background-color: #252525;
		}
		section[data-testid="stSidebar"] h1 {
			color: #ffffff !important;
			font-size: 1.5rem !important;
		}
		section[data-testid="stSidebar"] label {
			color: #e0e0e0 !important;
		}
		section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
			color: #e0e0e0 !important;
		}
		
		/* Divisores */
		hr {
			border-color: #333333 !important;
			margin: 2rem 0;
		}
		
		/* Botones multiselect */
		.stMultiSelect [data-baseweb="tag"] {
			background-color: #ef4444 !important;
			color: #ffffff !important;
			border: 1px solid #ef4444 !important;
		}
		
		.stMultiSelect [data-baseweb="tag"]:hover {
			background-color: #ef4444 !important;
		}
		
		/* Opciones del multiselect */
		[data-baseweb="popover"] {
			background-color: #252525 !important;
		}
		
		/* Items del dropdown */
		[data-baseweb="menu"] {
			background-color: #252525 !important;
		}
		
		[role="option"] {
			background-color: #252525 !important;
			color: #e0e0e0 !important;
		}
		
		[role="option"]:hover {
			background-color: #333333 !important;
		}
		
		/* Input del multiselect */
		.stMultiSelect > div > div {
			background-color: #2a2a2a !important;
			border-color: #3b82f6 !important;
		}
		
		/* Spinner */
		.stSpinner > div {
			border-top-color: #3b82f6 !important;
		}
		
		/* Alertas */
		.stAlert {
			background-color: #2a2a2a;
			border-left: 4px solid;
		}
		
		/* Inputs del sidebar */
		section[data-testid="stSidebar"] input {
			background-color: #1a1a1a !important;
			color: #ffffff !important;
		}
		</style>
	""", unsafe_allow_html=True)
	
	st.title("Panel COVID-19 Global")
	st.markdown("Visualización y análisis de datos epidemiológicos")
	st.markdown("---")

	# ========================================
	# CONFIGURACIÓN DE ORIGEN DE DATOS
	# ========================================
	
	# MEJOR OPCIÓN: Dropbox (más confiable que Google Drive)
	# 1. Sube tu archivo a Dropbox
	# 2. Click derecho → Compartir → Crear enlace
	# 3. Copia el link y CAMBIA ?dl=0 por ?dl=1 al final
	# Ejemplo: https://www.dropbox.com/s/abc123/archivo.csv.gz?dl=1
	
	USE_DROPBOX = False  # Cambia a True cuando tengas el link
	DROPBOX_URL = "https://www.dropbox.com/scl/fi/TU_LINK_AQUI/covid_2020_2022.csv.gz?rlkey=XXXX&dl=1"
	
	# Alternativa: GitHub Release (puede funcionar mejor ahora)
	USE_GITHUB = True
	GITHUB_URL = "https://github.com/AlvaroMolinaCL/ProyectoGestionDeDatos/releases/download/v1.0/covid_2020_2022.csv.gz"
	
	# Google Drive (problemático para archivos grandes)
	GOOGLE_DRIVE_FILE_ID = "185x7Rm2g04e304hc_rTLY1et-zymcByC"
	USE_GOOGLE_DRIVE = False
	
	# ========================================
	# CARGAR DATOS SEGÚN OPCIÓN SELECCIONADA
	# ========================================
	
	df_raw = None
	
	if USE_DROPBOX:
		st.info("📦 Cargando desde Dropbox...")
		df_raw = load_data_from_url_simple(DROPBOX_URL)
	elif USE_GITHUB:
		st.info("📦 Cargando desde GitHub Release...")
		df_raw = load_data_from_url_simple(GITHUB_URL)
	elif USE_GOOGLE_DRIVE:
		st.info("📦 Cargando desde Google Drive...")
		df_raw = load_data_from_google_drive(GOOGLE_DRIVE_FILE_ID)
	
	# Si falla, intentar con GitHub como respaldo
	if df_raw is None and not USE_GITHUB:
		st.warning("⚠️ Intentando cargar desde GitHub Release como alternativa...")
		df_raw = load_data_from_url_simple(GITHUB_URL)
	
	if df_raw is None:
		st.error("No se pueden cargar los datos. Verifica la conexión o el origen de datos.")
		return
	
	# Procesar dataframe
	df = process_dataframe(df_raw)

	if df is None or df.empty:
		st.error("Error al procesar los datos.")
		return

	# Filtros
	st.sidebar.title("Filtros")
	continents = sorted(df["continent"].dropna().unique())
	selected_continents = st.sidebar.multiselect(
		"Continente", 
		options=continents, 
		default=continents
	)

	# Si no hay continentes seleccionados, usar todos
	if not selected_continents:
		df_cont = df
	else:
		df_cont = df[df["continent"].isin(selected_continents)]

	countries = sorted(df_cont["country"].unique())
	selected_countries = st.sidebar.multiselect(
		"País", 
		options=countries, 
		default=countries[:10] if len(countries) > 10 else countries
	)

	# Si no hay países seleccionados, usar todos los disponibles
	if not selected_countries:
		df_country = df_cont
	else:
		df_country = df_cont[df_cont["country"].isin(selected_countries)]

	min_date = df_country["date"].min()
	max_date = df_country["date"].max()
	selected_date = st.sidebar.date_input(
		"Rango de fechas", 
		value=(min_date.date(), max_date.date())
	)
	if isinstance(selected_date, tuple) and len(selected_date) == 2:
		start_date, end_date = pd.to_datetime(selected_date[0]), pd.to_datetime(selected_date[1])
	else:
		start_date, end_date = min_date, max_date

	mask = (df_country["date"] >= start_date) & (df_country["date"] <= end_date)
	df_filtered = df_country.loc[mask].copy()

	if df_filtered.empty:
		st.warning("No hay datos disponibles para los filtros seleccionados.")
		return

	# Indicadores principales
	indicators, latest = compute_indicators(df_filtered)

	st.markdown("### Indicadores Clave")
	
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		st.metric(
			label="Casos Confirmados",
			value=f"{indicators['confirmed']:,}"
		)
	
	with col2:
		st.metric(
			label="Casos Activos",
			value=f"{indicators['active']:,}"
		)
	
	with col3:
		st.metric(
			label="Recuperados",
			value=f"{indicators['recovered']:,}"
		)
	
	with col4:
		st.metric(
			label="Fallecidos",
			value=f"{indicators['deaths']:,}"
		)

	# Gráficos
	st.markdown("---")
	st.markdown("### Evolución Temporal")
	
	timeseries = df_filtered.groupby("date")[["confirmed", "active", "recovered", "deaths"]].sum().reset_index()
	fig = px.line(
		timeseries, 
		x="date", 
		y=["confirmed", "active", "recovered", "deaths"],
		labels={"value":"Casos", "variable":"Indicador", "date":"Fecha"},
		color_discrete_map={
			"confirmed": "#ef4444",
			"active": "#f59e0b",
			"recovered": "#10b981",
			"deaths": "#6b7280"
		}
	)
	fig.update_layout(
		hovermode='x unified',
		plot_bgcolor='#2a2a2a',
		paper_bgcolor='#1a1a1a',
		font=dict(color='#e0e0e0'),
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1,
			font=dict(color='#e0e0e0')
		),
		xaxis=dict(
			gridcolor='#333333',
			showgrid=True,
			color='#e0e0e0'
		),
		yaxis=dict(
			gridcolor='#333333',
			showgrid=True,
			color='#e0e0e0'
		)
	)
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("### Nuevos Casos Diarios")
	new_cases = df_filtered.groupby("date")["new_confirmed"].sum().reset_index()
	fig2 = px.bar(
		new_cases.tail(60), 
		x="date", 
		y="new_confirmed",
		labels={"new_confirmed": "Nuevos Casos", "date": "Fecha"},
		color_discrete_sequence=["#3b82f6"]
	)
	fig2.update_layout(
		plot_bgcolor='#2a2a2a',
		paper_bgcolor='#1a1a1a',
		font=dict(color='#e0e0e0'),
		xaxis=dict(
			gridcolor='#333333',
			showgrid=True,
			color='#e0e0e0'
		),
		yaxis=dict(
			gridcolor='#333333',
			showgrid=True,
			color='#e0e0e0'
		)
	)
	st.plotly_chart(fig2, use_container_width=True)

	st.markdown("### Distribución por País")
	last_by_country = df_filtered.sort_values("date").groupby("country").last().reset_index()
	fig3 = px.bar(
		last_by_country.sort_values("confirmed", ascending=False).head(20),
		x="country", 
		y=["confirmed", "active", "recovered", "deaths"],
		labels={"value": "Casos", "country": "País"},
		color_discrete_map={
			"confirmed": "#ef4444",
			"active": "#f59e0b",
			"recovered": "#10b981",
			"deaths": "#6b7280"
		}
	)
	fig3.update_layout(
		plot_bgcolor='#2a2a2a',
		paper_bgcolor='#1a1a1a',
		font=dict(color='#e0e0e0'),
		xaxis_tickangle=-45,
		xaxis=dict(
			gridcolor='#333333',
			showgrid=False,
			color='#e0e0e0'
		),
		yaxis=dict(
			gridcolor='#333333',
			showgrid=True,
			color='#e0e0e0'
		),
		legend=dict(
			font=dict(color='#e0e0e0')
		)
	)
	st.plotly_chart(fig3, use_container_width=True)

	# Indicador de rebrote y tasa de crecimiento
	st.markdown("---")
	st.markdown("### Análisis de Crecimiento")
	
	total_series = df_filtered.groupby("date")["new_confirmed"].sum().sort_index()
	ratio, is_rebrote = rebrote_indicator(total_series.values)
	g_rate = growth_rate(df_filtered.groupby("date")["confirmed"].sum())

	rcol1, rcol2 = st.columns(2)
	rcol1.metric("Ratio de Rebrote (14d vs 14d previos)", f"{ratio:.2f}")
	rcol2.metric("Tasa de Crecimiento Diaria (promedio 7d)", f"{g_rate:.2f}%")
	
	if is_rebrote:
		st.error("Alerta: Los indicadores muestran señales de rebrote.")
	else:
		st.success("No se detecta rebrote significativo.")

	# Insights automáticos
	st.markdown("---")
	st.markdown("### Análisis Automático")
	insights = generate_insights(df_filtered)
	for insight in insights:
		st.markdown(f"- {insight}")


if __name__ == "__main__":
	main()