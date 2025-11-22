import os
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

@st.cache_data
def load_data(path="data/covid_data.csv"):
	"""Carga los datos desde `path`. Si no existe, genera un dataset de ejemplo."""
	if os.path.exists(path):
		df = pd.read_csv(path)
	else:
		# Generar dataset sintético de ejemplo
		dates = pd.date_range(end=pd.Timestamp.today(), periods=180)
		countries = [
			("Asia", "India"),
			("Asia", "China"),
			("Europe", "Spain"),
			("Europe", "Germany"),
			("America", "Chile"),
			("America", "USA"),
			("Africa", "Nigeria"),
		]
		rows = []
		for cont, country in countries:
			base = np.random.randint(1000, 20000)
			growth = np.random.uniform(0.98, 1.03)
			cases = base * (growth ** np.arange(len(dates)))
			noise = np.random.normal(0, cases * 0.02)
			confirmed = (cases + noise).clip(min=0).astype(int)
			recovered = (confirmed * np.random.uniform(0.6, 0.95)).astype(int)
			deaths = (confirmed * np.random.uniform(0.005, 0.03)).astype(int)
			active = (confirmed - recovered - deaths).clip(min=0).astype(int)
			for d, c, r, de, a in zip(dates, confirmed, recovered, deaths, active):
				rows.append(
					{
						"date": d.strftime("%Y-%m-%d"),
						"continent": cont,
						"country": country,
						"confirmed": int(c),
						"recovered": int(r),
						"deaths": int(de),
						"active": int(a),
					}
				)
		df = pd.DataFrame(rows)

	# Normalizar columnas y tipos
	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"])
	else:
		raise ValueError("El dataset debe contener una columna 'date'")

	# Asegurar columnas necesarias
	for col in ["confirmed", "recovered", "deaths"]:
		if col not in df.columns:
			df[col] = 0

	if "active" not in df.columns:
		df["active"] = df["confirmed"] - df["recovered"] - df["deaths"]

	# Orden
	df = df.sort_values(["country", "date"]).reset_index(drop=True)

	# Calcular nuevos casos diarios
	df["new_confirmed"] = df.groupby("country")["confirmed"].diff().fillna(0).clip(lower=0)
	return df


def compute_indicators(df):
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
	"""Calcula si hay rebrote comparando medias móviles recientes vs previas.

	Devuelve (ratio, is_rebrote:boolean)
	"""
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
	insights = []
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
	st.title("Dashboard de COVID — ProyectoGestiónDeDatos")

	st.sidebar.header("Carga y filtros")
	data_path = st.sidebar.text_input("Ruta CSV (opcional)", value="data/covid_data.csv")
	df = load_data(data_path)

	# Filtros
	continents = sorted(df["continent"].dropna().unique())
	selected_continents = st.sidebar.multiselect("Continente", options=continents, default=continents)

	df_cont = df[df["continent"].isin(selected_continents)]

	countries = sorted(df_cont["country"].unique())
	selected_countries = st.sidebar.multiselect("País", options=countries, default=countries)

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
		st.warning("No hay datos para los filtros seleccionados. Ajusta continente/país/fechas.")
		return

	# Indicadores principales
	indicators, latest = compute_indicators(df_filtered)

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Confirmados", f"{indicators['confirmed']:,}")
	col2.metric("Activos", f"{indicators['active']:,}")
	col3.metric("Recuperados", f"{indicators['recovered']:,}")
	col4.metric("Fallecidos", f"{indicators['deaths']:,}")

	st.markdown("**Gráficos temporales**")
	timeseries = df_filtered.groupby("date")[["confirmed", "active", "recovered", "deaths"]].sum().reset_index()
	fig = px.line(timeseries, x="date", y=["confirmed", "active", "recovered", "deaths"],
				  labels={"value":"Casos", "variable":"Indicador"},
				  title="Evolución temporal (agregado)")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("**Nuevos casos (últimos 60 días)**")
	new_cases = df_filtered.groupby("date")["new_confirmed"].sum().reset_index()
	fig2 = px.bar(new_cases.tail(60), x="date", y="new_confirmed", title="Nuevos casos diarios")
	st.plotly_chart(fig2, use_container_width=True)

	st.markdown("**Distribución por país (valor al final del rango)**")
	last_by_country = df_filtered.sort_values("date").groupby("country").last().reset_index()
	fig3 = px.bar(last_by_country.sort_values("confirmed", ascending=False).head(20),
				  x="country", y=["confirmed", "active", "recovered", "deaths"],
				  title="Top países — últimos valores")
	st.plotly_chart(fig3, use_container_width=True)

	# Indicador de rebrote y tasa de crecimiento
	total_series = df_filtered.groupby("date")["new_confirmed"].sum().sort_index()
	ratio, is_rebrote = rebrote_indicator(total_series.values)
	g_rate = growth_rate(df_filtered.groupby("date")["confirmed"].sum())

	st.markdown("**Indicadores de crecimiento**")
	rcol1, rcol2 = st.columns(2)
	rcol1.metric("Ratio rebrote (últimos 14 vs prev 14)", f"{ratio:.2f}", delta=None)
	rcol2.metric("Tasa de crecimiento diaria (media 7 días)", f"{g_rate:.2f}%")
	if is_rebrote:
		st.error("Alerta: indicadores muestran señales de rebrote.")
	else:
		st.success("No se detecta rebrote significativo según el umbral actual.")

	# Insights automáticos
	st.markdown("**Insights automáticos**")
	insights = generate_insights(df_filtered)
	for insight in insights:
		st.write("- ", insight)


if __name__ == "__main__":
	main()