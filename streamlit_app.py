import os
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

@st.cache_data
def load_data(path="data/covid_2020_2022.csv"):
	"""Carga los datos desde `path`. Maneja el formato real del CSV."""
	if not os.path.exists(path):
		st.error(f"Archivo no encontrado: {path}")
		st.info("Asegúrate de que el archivo 'covid_2020_2022.csv' esté en la carpeta 'data/'")
		return None
	
	# Leer CSV
	df = pd.read_csv(path)
	
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
		return ["No hay datos suficientes para generar insights."]
	
	# Casos nuevos agregados
	new_cases_total = int(df_filtered["new_confirmed"].sum())
	insights.append(f"Nuevos casos en rango seleccionado: {new_cases_total:,}")

	# Tendencia global (últimos 14 días)
	agg = df_filtered.groupby("date")["new_confirmed"].sum().sort_index()
	if len(agg) >= 14:
		last14 = agg[-14:].sum()
		prev14 = agg[-28:-14].sum() if len(agg) >= 28 else 0
		if prev14 == 0:
			insights.append("No hay periodo previo completo para comparar tendencia de 14 días.")
		else:
			change = (last14 - prev14) / prev14
			pct = change * 100
			sign = "aumento" if change > 0 else "disminución"
			insights.append(f"Comparado con 14 días previos: {abs(pct):.1f}% de {sign} en nuevos casos.")

	# País con mayor nuevos casos
	by_country = df_filtered.groupby("country")["new_confirmed"].sum()
	if not by_country.empty:
		top = by_country.idxmax()
		insights.append(f"País con más nuevos casos: {top} ({int(by_country.max()):,} casos)")

	return insights


def main():
	st.set_page_config(page_title="Dashboard COVID", layout="wide")
	st.title("📊 Dashboard de COVID — ProyectoGestiónDeDatos")

	st.sidebar.header("⚙️ Configuración")
	data_path = st.sidebar.text_input("Ruta CSV", value="data/covid_2020_2022.csv")
	
	# Cargar datos con mensaje de carga
	with st.spinner('Cargando datos...'):
		df = load_data(data_path)
	
	if df is None or df.empty:
		st.error("No se pudieron cargar los datos. Verifica la ruta del archivo.")
		return

	st.sidebar.success(f"✅ Datos cargados: {len(df):,} registros")

	# Filtros
	st.sidebar.header("🔍 Filtros")
	continents = sorted(df["continent"].dropna().unique())
	selected_continents = st.sidebar.multiselect("Continente", options=continents, default=continents)

	df_cont = df[df["continent"].isin(selected_continents)]

	countries = sorted(df_cont["country"].unique())
	selected_countries = st.sidebar.multiselect(
		"País", 
		options=countries, 
		default=countries[:10] if len(countries) > 10 else countries,
		help="Selecciona uno o más países"
	)

	df_country = df_cont[df_cont["country"].isin(selected_countries)]

	min_date = df_country["date"].min()
	max_date = df_country["date"].max()
	selected_date = st.sidebar.date_input("Rango de fechas", value=(min_date.date(), max_date.date()))
	if isinstance(selected_date, tuple) and len(selected_date) == 2:
		start_date, end_date = pd.to_datetime(selected_date[0]), pd.to_datetime(selected_date[1])
	else:
		start_date, end_date = min_date, max_date

	mask = (df_country["date"] >= start_date) & (df_country["date"] <= end_date)
	df_filtered = df_country.loc[mask].copy()

	if df_filtered.empty:
		st.warning("⚠️ No hay datos para los filtros seleccionados. Ajusta continente/país/fechas.")
		return

	# Indicadores principales
	indicators, latest = compute_indicators(df_filtered)

	st.markdown("### 📈 Indicadores Principales")
	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Confirmados", f"{indicators['confirmed']:,}")
	col2.metric("Activos", f"{indicators['active']:,}")
	col3.metric("Recuperados", f"{indicators['recovered']:,}")
	col4.metric("Fallecidos", f"{indicators['deaths']:,}")

	# Gráficos
	st.markdown("---")
	st.markdown("### 📊 Gráficos Temporales")
	
	timeseries = df_filtered.groupby("date")[["confirmed", "active", "recovered", "deaths"]].sum().reset_index()
	fig = px.line(timeseries, x="date", y=["confirmed", "active", "recovered", "deaths"],
				  labels={"value":"Casos", "variable":"Indicador"},
				  title="Evolución temporal (agregado)")
	fig.update_layout(hovermode='x unified')
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("### 📊 Nuevos casos (últimos 60 días)")
	new_cases = df_filtered.groupby("date")["new_confirmed"].sum().reset_index()
	fig2 = px.bar(new_cases.tail(60), x="date", y="new_confirmed", 
				  title="Nuevos casos diarios",
				  labels={"new_confirmed": "Nuevos casos", "date": "Fecha"})
	st.plotly_chart(fig2, use_container_width=True)

	st.markdown("### 🌍 Distribución por país")
	last_by_country = df_filtered.sort_values("date").groupby("country").last().reset_index()
	fig3 = px.bar(last_by_country.sort_values("confirmed", ascending=False).head(20),
				  x="country", y=["confirmed", "active", "recovered", "deaths"],
				  title="Top 20 países — últimos valores en el rango",
				  labels={"value": "Casos", "country": "País"})
	st.plotly_chart(fig3, use_container_width=True)

	# Indicador de rebrote y tasa de crecimiento
	st.markdown("---")
	st.markdown("### 🔬 Indicadores de Crecimiento")
	
	total_series = df_filtered.groupby("date")["new_confirmed"].sum().sort_index()
	ratio, is_rebrote = rebrote_indicator(total_series.values)
	g_rate = growth_rate(df_filtered.groupby("date")["confirmed"].sum())

	rcol1, rcol2 = st.columns(2)
	rcol1.metric("Ratio rebrote (últimos 14 vs prev 14)", f"{ratio:.2f}")
	rcol2.metric("Tasa de crecimiento diaria (media 7 días)", f"{g_rate:.2f}%")
	
	if is_rebrote:
		st.error("⚠️ Alerta: indicadores muestran señales de rebrote.")
	else:
		st.success("✅ No se detecta rebrote significativo según el umbral actual.")

	# Insights automáticos
	st.markdown("---")
	st.markdown("### 💡 Insights Automáticos")
	insights = generate_insights(df_filtered)
	for insight in insights:
		st.write("• ", insight)


if __name__ == "__main__":
	main()