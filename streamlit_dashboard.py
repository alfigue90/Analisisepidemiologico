import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
from scipy import stats
import plotly.graph_objs as go
import plotly.express as px
import io
import json
from datetime import datetime
import matplotlib.pyplot as plt
import base64
from PIL import Image

# Añadir la ruta del proyecto al PATH de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_and_preprocess_data
from src.interactive_visualizations import create_interactive_visualizations
from src.statistical_analysis import perform_advanced_analysis

def get_year_max_weeks(year):
    """
    Obtiene el número máximo de semanas para un año específico.
    """
    return 53 if year == 2025 else 52

def load_data():
    """
    Carga y preprocesa los datos de todos los años.
    """
    try:
        data_frames = []
        for year in range(2021, 2026):
            df = load_and_preprocess_data(f'data/{year}.txt', year, 'all')
            data_frames.append(df)
        return pd.concat(data_frames)
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

def perform_basic_analysis(df, analysis_type):
    """
    Realiza un análisis básico de los datos.
    """
    total_cases = len(df)
    avg_age = df['Edad'].mean()
    gender_distribution = df['Sexo'].value_counts(normalize=True) * 100
    
    analysis = {
        'Total de casos': total_cases,
        'Edad promedio': round(avg_age, 2),
        'Distribución por género': gender_distribution.to_dict()
    }
    
    if analysis_type in ['respiratorio', 'gastrointestinal']:
        most_common_diagnosis = df['Diagnostico Principal'].mode().values[0]
        analysis['Diagnóstico más común'] = most_common_diagnosis
    elif analysis_type in ['varicela', 'manopieboca']:
        severity = df[df['Destino'] != 'DOMICILIO'].shape[0] / total_cases * 100
        analysis['Porcentaje de casos severos'] = round(severity, 2)
    
    return analysis

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Dashboard de Análisis Epidemiológico",
        page_icon=":microscope:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Estilos CSS personalizados
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .reportview-container .main .block-container {
            max-width: 1200px;
        }
        h1, h2, h3 {
            color: #0e4194;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f0f0;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e4194;
            color: white;
        }
        div.stButton > button:first-child {
            background-color: #0e4194;
            color: white;
            border-radius: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .trend-up {
            color: red;
        }
        .trend-down {
            color: green;
        }
        .trend-neutral {
            color: gray;
        }
        .insight-box {
            background-color: #e7f0fd;
            border-left: 4px solid #0e4194;
            padding: 10px;
            margin: 10px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Título y descripción
    st.title('🔬 Dashboard de Análisis Epidemiológico')
    st.markdown("*SAR-SAPU San Pedro de la Paz | Sistema Avanzado de Vigilancia Epidemiológica*")
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        df = load_data()
    
    if df is None:
        st.error("No se pudieron cargar los datos. Por favor, verifica que los archivos existen y tienen el formato correcto.")
        st.stop()
    
    # Sidebar para filtros
    st.sidebar.header("Configuración")
    
    # Selector de tipo de análisis
    analysis_type = st.sidebar.selectbox(
        "Tipo de Análisis",
        options=['respiratorio', 'gastrointestinal', 'varicela', 'manopieboca'],
        format_func=lambda x: x.capitalize()
    )
    
    # Filtros de tiempo
    st.sidebar.subheader("Filtros Temporales")
    year_filter = st.sidebar.multiselect(
        "Años",
        options=sorted(df['Año'].unique()),
        default=sorted(df['Año'].unique())
    )
    
    # Otros filtros
    establecimientos_unicos = [e for e in df['Estableciemiento'].unique() if pd.notna(e)]
    establishments = st.sidebar.multiselect(
        "Establecimientos",
        options=sorted(establecimientos_unicos),
        default=sorted(establecimientos_unicos)
    )
    grupos_edad_unicos = [g for g in df['Grupo_Edad'].unique() if pd.notna(g)]
    age_groups = st.sidebar.multiselect(
        "Grupos de Edad",
        options=sorted(grupos_edad_unicos),
        default=sorted(grupos_edad_unicos)
    )
    
    # Filtrar datos según selección
    filtered_df = filter_data(df, analysis_type, year_filter, establishments, age_groups)
    
    if filtered_df.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        st.stop()
    
    # Ejecutar análisis avanzado
    with st.spinner('Realizando análisis avanzado... puede tardar unos segundos'):
        analysis_results = perform_advanced_analysis(filtered_df, analysis_type)
    
    # Organizar el dashboard en pestañas
    tabs = st.tabs(["📊 Resumen General", "🔎 Análisis Avanzado", "📈 Series Temporales", 
                    "🦠 Análisis Específico", "📝 Análisis GPT", "📋 Datos"])
    
    # Pestaña de Resumen General
    with tabs[0]:
        display_general_summary(filtered_df, analysis_results, analysis_type)
    
    # Pestaña de Análisis Avanzado
    with tabs[1]:
        display_advanced_analysis(filtered_df, analysis_results, analysis_type)
    
    # Pestaña de Series Temporales
    with tabs[2]:
        display_time_series_analysis(filtered_df, analysis_results, analysis_type)
    
    # Pestaña de Análisis Específico
    with tabs[3]:
        display_specific_analysis(filtered_df, analysis_results, analysis_type)
    
    # Pestaña de Análisis GPT
    with tabs[4]:
        display_gpt_analysis(analysis_results, analysis_type)
    
    # Pestaña de Datos
    with tabs[5]:
        display_raw_data(filtered_df)
    
    # Sección de exportación de datos y resultados
    st.sidebar.header("Exportar Resultados")
    
    export_format = st.sidebar.selectbox(
        "Formato de Exportación",
        options=["CSV", "Excel", "JSON", "PDF"]
    )
    
    if st.sidebar.button("Exportar Datos y Análisis"):
        export_data(filtered_df, analysis_results, analysis_type, export_format)
    
    # Información de la última actualización
    st.sidebar.markdown("---")
    st.sidebar.info(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.sidebar.markdown("Desarrollado por Equipo de Epidemiología")

def filter_data(df, analysis_type, year_filter, establishments, age_groups):
    """
    Filtra los datos según los criterios seleccionados.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_type (str): Tipo de análisis
        year_filter (list): Lista de años a incluir
        establishments (list): Lista de establecimientos a incluir
        age_groups (list): Lista de grupos de edad a incluir
        
    Returns:
        pandas.DataFrame: DataFrame filtrado
    """
    filtered_df = df.copy()
    
    # Aplicar filtros
    if year_filter:
        filtered_df = filtered_df[filtered_df['Año'].isin(year_filter)]
    
    if establishments:
        filtered_df = filtered_df[filtered_df['Estableciemiento'].isin(establishments)]
    
    if age_groups:
        filtered_df = filtered_df[filtered_df['Grupo_Edad'].isin(age_groups)]
    
    # Filtrar por tipo de análisis
    if analysis_type == 'respiratorio':
        filtered_df = filtered_df[filtered_df['CIE10 DP'].str.startswith('J', na=False) | 
                                  filtered_df['CIE10 DP'].str.startswith('U07', na=False)]
    elif analysis_type == 'gastrointestinal':
        filtered_df = filtered_df[filtered_df['CIE10 DP'].str.startswith('A', na=False)]
    elif analysis_type == 'varicela':
        filtered_df = filtered_df[filtered_df['CIE10 DP'] == 'B019']
    elif analysis_type == 'manopieboca':
        filtered_df = filtered_df[filtered_df['CIE10 DP'] == 'B084']
    
    return filtered_df

def display_general_summary(df, analysis_results, analysis_type):
    """
    Muestra el resumen general de los datos.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
        analysis_type (str): Tipo de análisis
    """
    st.header("📊 Resumen General")
    
    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Casos", f"{len(df):,}")
    
    with col2:
        avg_age = df['Edad_Anios'].mean()
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
    
    with col3:
        severity = len(df[df['Destino'] != 'DOMICILIO']) / len(df) * 100 if len(df) > 0 else 0
        st.metric("Tasa de Derivación", f"{severity:.1f}%")
    
    with col4:
        # Calcular tendencia porcentual comparando último y primer año
        years = sorted(df['Año'].unique())
        if len(years) >= 2:
            first_year = df[df['Año'] == years[0]].shape[0]
            last_year = df[df['Año'] == years[-1]].shape[0]
            if first_year > 0:
                trend_pct = (last_year - first_year) / first_year * 100
                trend_text = f"{trend_pct:+.1f}%"
                delta_color = "normal" if abs(trend_pct) < 10 else "inverse" if trend_pct < 0 else "normal"
                st.metric("Tendencia", f"{analysis_results.get('trend', 0):.2f}", delta=trend_text, delta_color=delta_color)
            else:
                st.metric("Tendencia", f"{analysis_results.get('trend', 0):.2f}")
        else:
            st.metric("Tendencia", f"{analysis_results.get('trend', 0):.2f}")
    
    # Distribución temporal
    st.subheader("Distribución Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Casos por año
        yearly_counts = df.groupby('Año').size().reset_index(name='Casos')
        fig_yearly = px.bar(
            yearly_counts, 
            x='Año', 
            y='Casos',
            title='Casos por Año',
            color='Casos',
            color_continuous_scale='Blues',
            text='Casos'
        )
        fig_yearly.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_yearly.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        # Casos por mes
        monthly_counts = df.groupby('Mes').size().reset_index(name='Casos')
        month_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
            7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        monthly_counts['Nombre Mes'] = monthly_counts['Mes'].map(month_names)
        fig_monthly = px.bar(
            monthly_counts, 
            x='Mes', 
            y='Casos',
            title='Casos por Mes',
            color='Casos',
            color_continuous_scale='Blues',
            labels={'Mes': 'Mes', 'Casos': 'Número de Casos'},
            text='Casos'
        )
        fig_monthly.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_monthly.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig_monthly.update_xaxes(tickvals=list(range(1, 13)), ticktext=list(month_names.values()))
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Distribución demográfica
    st.subheader("Distribución Demográfica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por género
        gender_counts = df['Sexo'].value_counts().reset_index()
        gender_counts.columns = ['Sexo', 'Casos']
        fig_gender = px.pie(
            gender_counts, 
            names='Sexo', 
            values='Casos',
            title='Distribución por Género',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Distribución por grupo de edad
        age_group_counts = df['Grupo_Edad'].value_counts().reset_index()
        age_group_counts.columns = ['Grupo de Edad', 'Casos']
        fig_age = px.pie(
            age_group_counts, 
            names='Grupo de Edad', 
            values='Casos',
            title='Distribución por Grupo de Edad',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_age.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Top diagnósticos
    st.subheader("Diagnósticos Más Frecuentes")
    
    top_diagnoses = df['Diagnostico Principal'].value_counts().head(10).reset_index()
    top_diagnoses.columns = ['Diagnóstico', 'Casos']
    
    fig_dx = px.bar(
        top_diagnoses, 
        y='Diagnóstico', 
        x='Casos',
        title='Top 10 Diagnósticos',
        orientation='h',
        color='Casos',
        color_continuous_scale='Blues',
        text='Casos'
    )
    fig_dx.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_dx.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_dx, use_container_width=True)
    
    # Principales Insights
    st.subheader("Principales Insights")
    
    # Extraer insights clave del análisis
    trend = analysis_results.get('trend', 0)
    trend_pvalue = analysis_results.get('trend_pvalue', 1)
    outbreaks = analysis_results.get('outbreaks', pd.DataFrame())
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Tendencia y Estacionalidad**")
        
        trend_direction = "creciente" if trend > 0 else "decreciente" if trend < 0 else "estable"
        trend_significance = "estadísticamente significativa" if trend_pvalue < 0.05 else "no estadísticamente significativa"
        
        st.markdown(f"* Tendencia: **{trend_direction}** ({trend:.2f} casos/semana, {trend_significance})")
        
        # Estacionalidad
        seasonal_component = analysis_results.get('seasonal_component')
        if seasonal_component is not None and hasattr(seasonal_component, 'seasonal'):
            seasonal_amplitude = seasonal_component.seasonal.max() - seasonal_component.seasonal.min()
            st.markdown(f"* Estacionalidad: **{seasonal_amplitude:.2f}** de amplitud")
        
        # Brotes
        st.markdown(f"* Brotes detectados: **{len(outbreaks)}**")
        
        # Ciclos
        cycles = enhanced_ts.get('epidemic_cycles', {})
        if cycles.get('cycle_detected', False):
            cycle_pattern = cycles.get('cycle_pattern', '')
            cycle_confidence = cycles.get('confidence', '')
            st.markdown(f"* Patrón cíclico: **{cycle_pattern.lower()}** (confianza: {cycle_confidence.lower()})")
            
            if 'next_outbreak_estimate' in cycles:
                next_outbreak = cycles['next_outbreak_estimate']
                if hasattr(next_outbreak, 'strftime'):
                    date_str = next_outbreak.strftime('%d/%m/%Y')
                    st.markdown(f"* Próximo brote estimado: **{date_str}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Métricas de Transmisibilidad y Severidad**")
        
        # Rt
        transmissibility = enhanced_ts.get('transmissibility_metrics', {})
        if 'current_rt' in transmissibility:
            rt = transmissibility['current_rt']
            rt_status = "CRECIMIENTO EPIDÉMICO" if rt > 1 else "DECRECIMIENTO EPIDÉMICO"
            st.markdown(f"* Número reproductivo efectivo (Rt): **{rt:.2f}** ({rt_status})")
        
        # Severidad
        severity_rate = len(df[df['Destino'] != 'DOMICILIO']) / len(df) * 100 if len(df) > 0 else 0
        severity_category = "alta" if severity_rate > 20 else "media" if severity_rate > 10 else "baja"
        st.markdown(f"* Tasa de derivación: **{severity_rate:.1f}%** (severidad {severity_category})")
        
        # Grupo más afectado
        most_affected = df['Grupo_Edad'].value_counts().idxmax()
        st.markdown(f"* Grupo etario más afectado: **{most_affected}**")
        
        # Pronóstico
        if 'forecast' in enhanced_ts and enhanced_ts['forecast'] is not None:
            forecast = enhanced_ts['forecast']
            if isinstance(forecast, pd.DataFrame) and not forecast.empty:
                forecast_value = forecast['forecast'].iloc[-1]
                st.markdown(f"* Pronóstico (próximas semanas): **{forecast_value:.1f}** casos")
        
        # Alertas
        early_warning = enhanced_ts.get('early_warning', {})
        if early_warning:
            historical_comp = early_warning.get('current_vs_historical', {})
            if 'status' in historical_comp:
                status = historical_comp['status']
                deviation = historical_comp.get('percent_deviation', 0)
                
                if status == "ALERT":
                    st.markdown(f"* ⚠️ **ALERTA**: Situación actual **{deviation:.1f}%** por encima del promedio histórico")
                elif status == "WARNING":
                    st.markdown(f"* ⚠️ **ADVERTENCIA**: Situación actual **{deviation:.1f}%** por encima del promedio histórico")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_advanced_analysis(df, analysis_results, analysis_type):
    """
    Muestra el análisis avanzado de los datos.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
        analysis_type (str): Tipo de análisis
    """
    st.header("🔎 Análisis Avanzado")
    
    # Análisis de tendencia y outliers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Análisis de Tendencia")
        
        # Mostrar tendencia
        trend = analysis_results.get('trend', 0)
        trend_pvalue = analysis_results.get('trend_pvalue', 1)
        ci = analysis_results.get('trend_confidence_interval', [0, 0])
        
        trend_direction = "creciente" if trend > 0 else "decreciente" if trend < 0 else "estable"
        trend_significance = "estadísticamente significativa" if trend_pvalue < 0.05 else "no estadísticamente significativa"
        
        st.markdown(f"**Tendencia {trend_direction}**: {trend:.4f} casos/semana")
        st.markdown(f"**Valor p**: {trend_pvalue:.4f} ({trend_significance})")
        st.markdown(f"**Intervalo de confianza (95%)**: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        # Visualización de tendencia
        df_weekly = analysis_results.get('df_weekly')
        if df_weekly is not None:
            fig = px.scatter(
                df_weekly.reset_index(), 
                x='fecha', 
                y='Casos',
                title='Tendencia de Casos',
                trendline='ols'
            )
            fig.update_layout(xaxis_title='Fecha', yaxis_title='Número de Casos')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Detección de Outliers")
        
        # Mostrar outliers
        outliers = analysis_results.get('outliers', pd.DataFrame())
        
        if not outliers.empty:
            st.markdown(f"**Número de outliers detectados**: {len(outliers)}")
            
            # Crear visualización de outliers
            df_weekly = analysis_results.get('df_weekly')
            if df_weekly is not None:
                fig = go.Figure()
                
                # Datos normales
                fig.add_trace(go.Scatter(
                    x=df_weekly.index,
                    y=df_weekly['Casos'],
                    mode='lines+markers',
                    name='Casos',
                    marker=dict(color='blue', size=6)
                ))
                
                # Outliers
                fig.add_trace(go.Scatter(
                    x=outliers.index,
                    y=outliers['Casos'],
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=12, symbol='star')
                ))
                
                fig.update_layout(
                    title='Detección de Outliers',
                    xaxis_title='Fecha',
                    yaxis_title='Número de Casos'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de outliers
            if st.checkbox("Mostrar tabla de outliers"):
                outliers_table = outliers.reset_index()
                outliers_table.columns = ['Fecha', 'Casos', 'Edad Promedio']
                st.dataframe(outliers_table)
        else:
            st.markdown("No se detectaron outliers.")
    
    # Análisis de autocorrelación
    st.subheader("Análisis de Autocorrelación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ACF
        acf_values = analysis_results.get('acf', [])
        if len(acf_values) > 0:
            fig = go.Figure()
            
            lag_values = list(range(len(acf_values)))
            fig.add_trace(go.Bar(
                x=lag_values,
                y=acf_values,
                name='ACF',
                marker_color='blue'
            ))
            
            # Añadir líneas para niveles de significancia (±1.96/√n)
            if len(df_weekly) > 0:
                significance_level = 1.96 / np.sqrt(len(df_weekly))
                fig.add_shape(
                    type="line",
                    x0=0, y0=significance_level,
                    x1=len(acf_values)-1, y1=significance_level,
                    line=dict(color="red", width=1, dash="dash"),
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=-significance_level,
                    x1=len(acf_values)-1, y1=-significance_level,
                    line=dict(color="red", width=1, dash="dash"),
                )
                
                # Área entre las líneas de significancia
                x_vals = list(range(len(acf_values))) + list(range(len(acf_values)-1, -1, -1))
                y_vals = [significance_level] * len(acf_values) + [-significance_level] * len(acf_values)
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(color="rgba(255, 0, 0, 0)"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            
            fig.update_layout(
                title='Función de Autocorrelación (ACF)',
                xaxis_title='Lag',
                yaxis_title='Autocorrelación',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretación
            significant_lags = [i for i, val in enumerate(acf_values) if i > 0 and abs(val) > 1.96 / np.sqrt(len(df_weekly))]
            
            if significant_lags:
                st.markdown(f"**Lags significativos**: {', '.join(map(str, significant_lags))}")
                if 52 in significant_lags or 26 in significant_lags or 13 in significant_lags or 12 in significant_lags:
                    st.markdown("⚠️ Se detecta posible estacionalidad (lags significativos en múltiplos de semanas/meses).")
            else:
                st.markdown("No se detectaron lags significativos en la autocorrelación.")
                
    with col2:
        # PACF
        pacf_values = analysis_results.get('pacf', [])
        if len(pacf_values) > 0:
            fig = go.Figure()
            
            lag_values = list(range(len(pacf_values)))
            fig.add_trace(go.Bar(
                x=lag_values,
                y=pacf_values,
                name='PACF',
                marker_color='green'
            ))
            
            # Añadir líneas para niveles de significancia (±1.96/√n)
            if len(df_weekly) > 0:
                significance_level = 1.96 / np.sqrt(len(df_weekly))
                fig.add_shape(
                    type="line",
                    x0=0, y0=significance_level,
                    x1=len(pacf_values)-1, y1=significance_level,
                    line=dict(color="red", width=1, dash="dash"),
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=-significance_level,
                    x1=len(pacf_values)-1, y1=-significance_level,
                    line=dict(color="red", width=1, dash="dash"),
                )
                
                # Área entre las líneas de significancia
                x_vals = list(range(len(pacf_values))) + list(range(len(pacf_values)-1, -1, -1))
                y_vals = [significance_level] * len(pacf_values) + [-significance_level] * len(pacf_values)
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(color="rgba(255, 0, 0, 0)"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            
            fig.update_layout(
                title='Función de Autocorrelación Parcial (PACF)',
                xaxis_title='Lag',
                yaxis_title='Autocorrelación Parcial',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretación
            significant_lags = [i for i, val in enumerate(pacf_values) if i > 0 and abs(val) > 1.96 / np.sqrt(len(df_weekly))]
            
            if significant_lags:
                st.markdown(f"**Lags significativos**: {', '.join(map(str, significant_lags))}")
                st.markdown("Los lags significativos en PACF sugieren los términos AR(p) para modelado ARIMA.")
            else:
                st.markdown("No se detectaron lags significativos en la autocorrelación parcial.")
    
    # Análisis por grupos
    st.subheader("Análisis por Grupos")
    
    group_tab1, group_tab2, group_tab3 = st.tabs(["Por Edad", "Por Género", "Por Establecimiento"])
    
    with group_tab1:
        age_analysis = analyze_age_groups(df)
        display_age_group_analysis(age_analysis)
    
    with group_tab2:
        gender_analysis = analyze_gender(df)
        display_gender_analysis(gender_analysis)
    
    with group_tab3:
        establishment_analysis = analyze_establishments(df)
        display_establishment_analysis(establishment_analysis)

def display_time_series_analysis(df, analysis_results, analysis_type):
    """
    Muestra el análisis avanzado de series temporales.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
        analysis_type (str): Tipo de análisis
    """
    st.header("📈 Análisis Avanzado de Series Temporales")
    
    # Obtener resultados del análisis avanzado de series temporales
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    
    if not enhanced_ts:
        st.warning("No hay resultados de análisis avanzado de series temporales disponibles.")
        return
    
    # Navegación por pestañas para diferentes aspectos del análisis
    ts_tabs = st.tabs([
        "Cambios Estructurales", 
        "Modelado ARIMA", 
        "Ciclos Epidémicos",
        "Transmisibilidad (Rt)"
    ])
    
    # Tab 1: Cambios Estructurales
    with ts_tabs[0]:
        st.subheader("Detección de Cambios Estructurales")
        
        structural_changes = enhanced_ts.get('structural_changes', {})
        change_points = structural_changes.get('change_points', [])
        
        if change_points:
            st.markdown(f"**Número de cambios detectados**: {len(change_points)}")
            
            # Visualización de cambios estructurales
            df_weekly = analysis_results.get('df_weekly')
            if df_weekly is not None:
                fig = go.Figure()
                
                # Datos originales
                fig.add_trace(go.Scatter(
                    x=df_weekly.index,
                    y=df_weekly['Casos'],
                    mode='lines',
                    name='Casos',
                    line=dict(color='blue', width=2)
                ))
                
                # Marcar puntos de cambio
                for cp in change_points:
                    if cp < len(df_weekly):
                        fig.add_shape(
                            type="line",
                            x0=df_weekly.index[cp], 
                            y0=0,
                            x1=df_weekly.index[cp], 
                            y1=df_weekly['Casos'].max() * 1.1,
                            line=dict(color="red", width=2, dash="dash"),
                        )
                
                # Añadir estadísticas de segmentos
                if 'segment_stats' in structural_changes:
                    for stat in structural_changes['segment_stats']:
                        if 'change_point' in stat and stat['change_point'] < len(df_weekly):
                            cp = stat['change_point']
                            fig.add_annotation(
                                x=df_weekly.index[cp],
                                y=df_weekly['Casos'].max() * 0.9,
                                text=f"{stat.get('relative_change', 0):.1%}",
                                showarrow=True,
                                arrowhead=1,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="red",
                                ax=0,
                                ay=-40
                            )
                
                fig.update_layout(
                    title='Detección de Cambios Estructurales en la Serie Temporal',
                    xaxis_title='Fecha',
                    yaxis_title='Número de Casos',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de estadísticas de segmentos
                if 'segment_stats' in structural_changes and st.checkbox("Mostrar estadísticas de segmentos"):
                    segment_stats = []
                    for stat in structural_changes['segment_stats']:
                        if 'change_point' in stat:
                            cp = stat['change_point']
                            if cp < len(df_weekly):
                                segment_stats.append({
                                    'Fecha': df_weekly.index[cp].strftime('%Y-%m-%d'),
                                    'Cambio Relativo': f"{stat.get('relative_change', 0):.1%}",
                                    'Media Previa': round(stat.get('pre_mean', 0), 2),
                                    'Media Posterior': round(stat.get('post_mean', 0), 2),
                                    'Valor p': round(stat.get('p_value', 1), 4),
                                    'Significativo': 'Sí' if stat.get('p_value', 1) < 0.05 else 'No'
                                })
                    
                    if segment_stats:
                        st.table(pd.DataFrame(segment_stats))
            
        else:
            st.markdown("**No se detectaron cambios estructurales en la serie temporal.**")
        
        # Interpretación
        with st.expander("Interpretación de Cambios Estructurales"):
            st.markdown("""
            ### ¿Qué son los cambios estructurales?
            
            Los cambios estructurales representan puntos en la serie temporal donde hay alteraciones significativas en el patrón 
            o comportamiento de los datos. Estos pueden indicar:
            
            - Inicio o fin de un brote epidémico
            - Cambios en la política de salud pública
            - Introducción de medidas de control
            - Cambios en el sistema de vigilancia o reporte
            
            ### Cómo interpretar los resultados
            
            - **Cambio relativo**: Indica la magnitud y dirección del cambio en la media de casos
            - **Significancia**: Un valor p < 0.05 sugiere que el cambio es estadísticamente significativo
            - **Patrones temporales**: Múltiples cambios en corto tiempo pueden indicar inestabilidad en la transmisión
            """)
    
    # Tab 2: Modelado ARIMA
    with ts_tabs[1]:
        st.subheader("Modelado ARIMA")
        
        # Información del modelo
        model_info = enhanced_ts.get('model_info', {})
        if model_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Características del Modelo**")
                st.markdown(f"- Orden ARIMA: {model_info.get('order', 'No especificado')}")
                st.markdown(f"- Orden estacional: {model_info.get('seasonal_order', 'No especificado')}")
                
                # Métricas de ajuste
                model_fit = enhanced_ts.get('model_fit', {})
                if model_fit:
                    st.markdown("**Métricas de Ajuste**")
                    st.markdown(f"- RMSE: {model_fit.get('rmse', 'No disponible'):.4f}")
                    st.markdown(f"- MAPE: {model_fit.get('mape', 'No disponible'):.2f}%")
            
            with col2:
                # Criterios de información
                st.markdown("**Criterios de Información**")
                st.markdown(f"- AIC: {model_info.get('aic', 'No disponible'):.2f}")
                st.markdown(f"- BIC: {model_info.get('bic', 'No disponible'):.2f}")
                
                # Estacionalidad detectada
                seasonality = enhanced_ts.get('detected_seasonality', {})
                if seasonality:
                    st.markdown("**Estacionalidad Detectada**")
                    st.markdown(f"- Período: {seasonality.get('period', 'No detectado')}")
        
        # Visualización del pronóstico
        forecast = enhanced_ts.get('forecast')
        if forecast is not None and isinstance(forecast, pd.DataFrame) and not forecast.empty:
            df_weekly = analysis_results.get('df_weekly')
            
            fig = go.Figure()
            
            # Datos históricos
            fig.add_trace(go.Scatter(
                x=df_weekly.index,
                y=df_weekly['Casos'],
                mode='lines',
                name='Datos históricos',
                line=dict(color='blue', width=2)
            ))
            
            # Pronóstico
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['forecast'],
                mode='lines',
                name='Pronóstico',
                line=dict(color='red', width=2)
            ))
            
            # Intervalo de confianza
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['upper_ci'],
                mode='lines',
                name='Límite superior (95%)',
                line=dict(color='rgba(255, 0, 0, 0.2)', width=0)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['lower_ci'],
                mode='lines',
                name='Límite inferior (95%)',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 0, 0, 0.2)', width=0)
            ))
            
            fig.update_layout(
                title='Pronóstico ARIMA',
                xaxis_title='Fecha',
                yaxis_title='Número de Casos',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de pronóstico
            if st.checkbox("Mostrar tabla de pronóstico"):
                forecast_table = forecast.copy()
                forecast_table.index = forecast_table.index.strftime('%Y-%m-%d')
                forecast_table.columns = ['Pronóstico', 'Límite Inferior', 'Límite Superior']
                forecast_table = forecast_table.round(1)
                st.dataframe(forecast_table)
        else:
            st.warning("No hay datos de pronóstico disponibles.")
        
        # Interpretación
        with st.expander("Interpretación del Modelado ARIMA"):
            st.markdown("""
            ### Modelado ARIMA
            
            El modelado ARIMA (AutoRegressive Integrated Moving Average) es una técnica estadística para analizar 
            y pronosticar series temporales. El modelo se especifica como ARIMA(p,d,q) donde:
            
            - **p**: Orden autoregresivo (número de observaciones pasadas que influyen en el valor actual)
            - **d**: Grado de diferenciación necesario para hacer la serie estacionaria
            - **q**: Orden de media móvil (número de términos de error pasados que influyen en el valor actual)
            
            Para series estacionales, se especifica como ARIMA(p,d,q)(P,D,Q,s) donde s es el período estacional.
            
            ### Métricas de Ajuste
            
            - **RMSE** (Root Mean Square Error): Medida del error promedio, valores más bajos indican mejor ajuste
            - **MAPE** (Mean Absolute Percentage Error): Error porcentual promedio, valores más bajos son mejores
            
            ### Interpretación del Pronóstico
            
            El pronóstico muestra la proyección esperada de casos para las próximas semanas, junto con un 
            intervalo de confianza del 95% que representa la incertidumbre del pronóstico.
            """)
    
    # Tab 3: Ciclos Epidémicos
    with ts_tabs[2]:
        st.subheader("Análisis de Ciclos Epidémicos")
        
        cycles = enhanced_ts.get('epidemic_cycles', {})
        
        if cycles.get('cycle_detected', False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Características del Ciclo**")
                
                pattern = cycles.get('cycle_pattern', '')
                pattern_desc = {
                    'REGULAR': 'Regular',
                    'MODERATELY_REGULAR': 'Moderadamente regular',
                    'IRREGULAR': 'Irregular',
                    'INSUFFICIENT_DATA': 'Datos insuficientes'
                }
                
                confidence = cycles.get('confidence', '')
                conf_desc = {
                    'HIGH': 'Alta',
                    'MEDIUM': 'Media',
                    'LOW': 'Baja',
                    'VERY_LOW': 'Muy baja'
                }
                
                st.markdown(f"- Patrón del ciclo: **{pattern_desc.get(pattern, pattern)}**")
                st.markdown(f"- Confianza: **{conf_desc.get(confidence, confidence)}**")
                
                # Intervalos entre brotes
                intervals = cycles.get('intervals', {})
                if intervals:
                    st.markdown(f"- Intervalo medio: **{intervals.get('mean', 0):.1f}** días")
                    st.markdown(f"- Intervalo mediano: **{intervals.get('median', 0):.1f}** días")
                    st.markdown(f"- Desviación estándar: **{intervals.get('std', 0):.1f}** días")
                    st.markdown(f"- Coeficiente de variación: **{intervals.get('cv', 0):.2f}**")
            
            with col2:
                st.markdown("**Predicción de Próximo Brote**")
                
                # Próximo brote
                if 'next_outbreak_estimate' in cycles:
                    next_outbreak = cycles['next_outbreak_estimate']
                    if hasattr(next_outbreak, 'strftime'):
                        next_date = next_outbreak.strftime('%d/%m/%Y')
                        st.markdown(f"- Fecha estimada: **{next_date}**")
                        
                        # Intervalo de predicción
                        if 'next_outbreak_interval' in cycles:
                            interval = cycles['next_outbreak_interval']
                            lower = interval['lower'].strftime('%d/%m/%Y') if hasattr(interval['lower'], 'strftime') else str(interval['lower'])
                            upper = interval['upper'].strftime('%d/%m/%Y') if hasattr(interval['upper'], 'strftime') else str(interval['upper'])
                            st.markdown(f"- Intervalo de predicción: **{lower}** - **{upper}**")
                
                # Estacionalidad
                if 'seasonality' in cycles and cycles['seasonality'].get('detected', False):
                    seasonality = cycles['seasonality']
                    
                    st.markdown("**Estacionalidad Detectada**")
                    
                    month_names = {
                        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                    }
                    
                    dominant_month = seasonality.get('dominant_month', 0)
                    month_name = month_names.get(dominant_month, str(dominant_month))
                    
                    st.markdown(f"- Mes dominante: **{month_name}**")
                    
                    if 'seasonal_period' in seasonality:
                        period = seasonality['seasonal_period']
                        period_months = [month_names.get(m, str(m)) for m in period]
                        st.markdown(f"- Período estacional: **{', '.join(period_months)}**")
            
            # Visualización de brotes y ciclos
            if 'n_outbreaks' in cycles and cycles['n_outbreaks'] > 0:
                df_weekly = analysis_results.get('df_weekly')
                outbreaks = analysis_results.get('outbreaks', pd.DataFrame())
                
                if not outbreaks.empty and df_weekly is not None:
                    fig = go.Figure()
                    
                    # Datos originales
                    fig.add_trace(go.Scatter(
                        x=df_weekly.index,
                        y=df_weekly['Casos'],
                        mode='lines',
                        name='Casos',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Brotes detectados
                    fig.add_trace(go.Scatter(
                        x=outbreaks.index,
                        y=outbreaks['Casos'],
                        mode='markers',
                        name='Brotes detectados',
                        marker=dict(color='red', size=12, symbol='star')
                    ))
                    
                    # Próximo brote estimado
                    if 'next_outbreak_estimate' in cycles:
                        next_outbreak = cycles['next_outbreak_estimate']
                        
                        if hasattr(next_outbreak, 'strftime'):
                            # Punto del próximo brote
                            fig.add_trace(go.Scatter(
                                x=[next_outbreak],
                                y=[df_weekly['Casos'].mean()],
                                mode='markers',
                                name='Próximo brote estimado',
                                marker=dict(color='orange', size=15, symbol='diamond')
                            ))
                            
                            # Línea vertical para el próximo brote
                            fig.add_shape(
                                type="line",
                                x0=next_outbreak, 
                                y0=0,
                                x1=next_outbreak, 
                                y1=df_weekly['Casos'].max() * 1.1,
                                line=dict(color="orange", width=2, dash="dash"),
                            )
                            
                            # Intervalo de predicción
                            if 'next_outbreak_interval' in cycles:
                                interval = cycles['next_outbreak_interval']
                                
                                if hasattr(interval['lower'], 'strftime') and hasattr(interval['upper'], 'strftime'):
                                    fig.add_shape(
                                        type="rect",
                                        x0=interval['lower'],
                                        x1=interval['upper'],
                                        y0=0,
                                        y1=df_weekly['Casos'].max() * 1.1,
                                        fillcolor="rgba(255, 165, 0, 0.1)",
                                        line=dict(color="rgba(255, 165, 0, 0.5)", width=1)
                                    )
                    
                    fig.update_layout(
                        title='Ciclos Epidémicos y Próximo Brote Estimado',
                        xaxis_title='Fecha',
                        yaxis_title='Número de Casos',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**No se detectó un patrón cíclico claro en los datos.**")
            
            if 'n_outbreaks' in cycles:
                st.markdown(f"Número de brotes detectados: {cycles['n_outbreaks']}")
                st.markdown("Insuficientes brotes para establecer un patrón cíclico confiable.")
        
        # Interpretación
        with st.expander("Interpretación de Ciclos Epidémicos"):
            st.markdown("""
            ### Ciclos Epidémicos
            
            Los ciclos epidémicos representan patrones recurrentes en la aparición de casos. Su identificación 
            permite anticipar futuros brotes y planificar intervenciones.
            
            ### Patrón del Ciclo
            
            - **Regular**: Intervalos consistentes entre brotes, alta predictibilidad
            - **Moderadamente regular**: Cierta variabilidad en los intervalos, predictibilidad moderada
            - **Irregular**: Alta variabilidad en los intervalos, baja predictibilidad
            
            ### Coeficiente de Variación (CV)
            
            El CV indica la consistencia del ciclo:
            - CV < 0.2: Ciclo regular
            - CV entre 0.2 y 0.4: Ciclo moderadamente regular
            - CV > 0.4: Ciclo irregular
            
            ### Confianza de la Predicción
            
            La confianza de la predicción del próximo brote depende de:
            - Regularidad del patrón cíclico
            - Número de brotes previos observados
            - Estabilidad de los intervalos entre brotes
            """)
    
    # Tab 4: Transmisibilidad (Rt)
    with ts_tabs[3]:
        st.subheader("Métricas de Transmisibilidad (Rt)")
        
        transmissibility = enhanced_ts.get('transmissibility_metrics', {})
        
        if 'error' not in transmissibility and 'rt_proxy' in transmissibility:
            col1, col2 = st.columns(2)
            
            with col1:
                current_rt = transmissibility.get('current_rt')
                
                if current_rt is not None:
                    # Indicador visual de Rt
                    if current_rt > 1:
                        st.markdown(f"<h3 style='color:red'>Rt = {current_rt:.2f}</h3>", unsafe_allow_html=True)
                        st.markdown("**Estado: CRECIMIENTO EPIDÉMICO**")
                    else:
                        st.markdown(f"<h3 style='color:green'>Rt = {current_rt:.2f}</h3>", unsafe_allow_html=True)
                        st.markdown("**Estado: DECRECIMIENTO EPIDÉMICO**")
                
                # Métricas adicionales
                rt_mean = transmissibility.get('recent_rt_mean')
                if rt_mean is not None:
                    st.markdown(f"Rt promedio reciente: {rt_mean:.2f}")
                
                rt_std = transmissibility.get('recent_rt_std')
                if rt_std is not None:
                    st.markdown(f"Desviación estándar de Rt: {rt_std:.2f}")
                
                above_threshold = transmissibility.get('above_threshold')
                if above_threshold is not None:
                    st.markdown(f"Proporción de tiempo con Rt>1: {above_threshold:.0%}")
            
            with col2:
                # Interpretación básica de Rt
                st.markdown("### Interpretación del Número Reproductivo Efectivo (Rt)")
                
                st.markdown("""
                - **Rt > 1**: Cada caso genera más de un caso nuevo en promedio → Crecimiento epidémico
                - **Rt = 1**: Cada caso genera exactamente un caso nuevo → Estabilidad epidémica
                - **Rt < 1**: Cada caso genera menos de un caso nuevo → Decrecimiento epidémico
                """)
                
                # Recomendaciones básicas según Rt
                if current_rt is not None:
                    st.markdown("### Implicaciones Epidemiológicas")
                    
                    if current_rt > 1.5:
                        st.markdown("""
                        **Crecimiento rápido (Rt > 1.5)**
                        - Posible inicio de un brote significativo
                        - Se recomiendan medidas de intervención inmediatas
                        - Intensificar vigilancia y seguimiento de casos
                        """)
                    elif current_rt > 1:
                        st.markdown("""
                        **Crecimiento moderado (1 < Rt ≤ 1.5)**
                        - Crecimiento sostenido de casos
                        - Considerar reforzar medidas preventivas
                        - Monitoreo estrecho de la evolución de casos
                        """)
                    elif current_rt > 0.5:
                        st.markdown("""
                        **Decrecimiento moderado (0.5 < Rt < 1)**
                        - Disminución gradual de casos
                        - Mantener medidas preventivas básicas
                        - Continuar la vigilancia habitual
                        """)
                    else:
                        st.markdown("""
                        **Decrecimiento rápido (Rt ≤ 0.5)**
                        - Enfermedad en fase de control
                        - Posible fin de brote epidémico
                        - Oportunidad para evaluación retrospectiva
                        """)
            
            # Visualización de Rt
            rt_proxy = transmissibility.get('rt_proxy')
            
            if rt_proxy:
                # Convertir a DataFrame para graficar
                if isinstance(rt_proxy, dict):
                    dates = []
                    values = []
                    
                    for date_str, value in rt_proxy.items():
                        try:
                            if isinstance(date_str, str):
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                            else:
                                date = date_str
                            
                            dates.append(date)
                            values.append(value)
                        except ValueError:
                            continue
                    
                    rt_df = pd.DataFrame({'date': dates, 'rt': values})
                    rt_df = rt_df.sort_values('date')
                    rt_df = rt_df.set_index('date')
                else:
                    rt_df = pd.DataFrame({'rt': rt_proxy})
                
                fig = go.Figure()
                
                # Línea de Rt
                fig.add_trace(go.Scatter(
                    x=rt_df.index,
                    y=rt_df['rt'],
                    mode='lines',
                    name='Rt',
                    line=dict(color='purple', width=2)
                ))
                
                # Línea horizontal en Rt=1
                fig.add_shape(
                    type="line",
                    x0=rt_df.index.min(), 
                    y0=1,
                    x1=rt_df.index.max(), 
                    y1=1,
                    line=dict(color="red", width=1, dash="dash"),
                )
                
                # Área de crecimiento (Rt > 1)
                fig.add_trace(go.Scatter(
                    x=rt_df.index,
                    y=rt_df['rt'].clip(lower=1),
                    fill='tonexty',
                    mode='none',
                    name='Crecimiento (Rt > 1)',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                ))
                
                # Área de decrecimiento (Rt < 1)
                fig.add_trace(go.Scatter(
                    x=rt_df.index,
                    y=rt_df['rt'].clip(upper=1),
                    fill='tonexty',
                    mode='none',
                    name='Decrecimiento (Rt < 1)',
                    fillcolor='rgba(0, 128, 0, 0.2)'
                ))
                
                fig.update_layout(
                    title='Número Reproductivo Efectivo (Rt) a lo largo del tiempo',
                    xaxis_title='Fecha',
                    yaxis_title='Rt',
                    yaxis=dict(range=[0, max(3, rt_df['rt'].max() * 1.1)]),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de valores Rt recientes
                if st.checkbox("Mostrar valores recientes de Rt"):
                    recent_rt = rt_df.tail(8).copy()
                    recent_rt.index = recent_rt.index.strftime('%Y-%m-%d')
                    recent_rt.columns = ['Rt']
                    st.dataframe(recent_rt)
        else:
            if 'error' in transmissibility:
                st.warning(f"Error en el cálculo de métricas de transmisibilidad: {transmissibility['error']}")
            else:
                st.warning("No hay datos disponibles para el cálculo de métricas de transmisibilidad.")
        
        # Interpretación
        with st.expander("Interpretación del Número Reproductivo Efectivo (Rt)"):
            st.markdown("""
            ### Número Reproductivo Efectivo (Rt)
            
            El número reproductivo efectivo (Rt) representa el promedio de casos secundarios generados por cada caso 
            infeccioso en un momento dado, considerando la inmunidad de la población y las medidas de control.
            
            ### Significado de Rt
            
            - **Rt > 1**: La enfermedad se está propagando, cada caso infecta a más de una persona
            - **Rt = 1**: La enfermedad está en equilibrio, cada caso infecta exactamente a una persona
            - **Rt < 1**: La enfermedad está en declive, cada caso infecta a menos de una persona
            
            ### Importancia para la Salud Pública
            
            Rt es un indicador clave para:
            
            - Monitorizar la transmisibilidad en tiempo real
            - Evaluar la efectividad de intervenciones
            - Predecir la trayectoria a corto plazo de la epidemia
            - Determinar si se necesitan medidas adicionales de control
            
            ### Limitaciones
            
            - Es una estimación basada en datos observados
            - Sensible a retrasos en notificación y subdiagnóstico
            - No captura completamente la heterogeneidad de transmisión
            """)

def display_specific_analysis(df, analysis_results, analysis_type):
    """
    Muestra el análisis específico según el tipo de enfermedad.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
        analysis_type (str): Tipo de análisis
    """
    st.header(f"🦠 Análisis Específico: {analysis_type.capitalize()}")
    
    if analysis_type == 'respiratorio':
        display_respiratory_analysis(df, analysis_results)
    elif analysis_type == 'gastrointestinal':
        display_gastrointestinal_analysis(df, analysis_results)
    elif analysis_type == 'varicela':
        display_varicela_analysis(df, analysis_results)
    elif analysis_type == 'manopieboca':
        display_manopieboca_analysis(df, analysis_results)

def display_respiratory_analysis(df, analysis_results):
    """
    Muestra el análisis específico para enfermedades respiratorias.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
    """
    # Análisis específico para enfermedades respiratorias
    resp_results = analysis_results.get('respiratorio_specific', {})
    
    # Top tipos de infecciones respiratorias
    st.subheader("Distribución de Infecciones Respiratorias")
    
    # Agrupar por primeros 3 caracteres de CIE10
    respiratory_types = df['CIE10 DP'].str[:3].value_counts().reset_index()
    respiratory_types.columns = ['Código', 'Casos']
    
    # Añadir descripciones
    code_descriptions = {
        'J00': 'Rinofaringitis aguda',
        'J01': 'Sinusitis aguda',
        'J02': 'Faringitis aguda',
        'J03': 'Amigdalitis aguda',
        'J04': 'Laringitis/traqueítis aguda',
        'J05': 'Laringitis/traqueítis aguda',
        'J06': 'Infecciones agudas VRA',
        'J09': 'Influenza por virus identificado',
        'J10': 'Influenza por virus identificado',
        'J11': 'Influenza, virus no identificado',
        'J12': 'Neumonía viral',
        'J13': 'Neumonía por S. pneumoniae',
        'J14': 'Neumonía por H. influenzae',
        'J15': 'Neumonía bacteriana',
        'J16': 'Neumonía por otros org. infecciosos',
        'J17': 'Neumonía en enf. clasificadas',
        'J18': 'Neumonía, org. no especificado',
        'J20': 'Bronquitis aguda',
        'J21': 'Bronquiolitis aguda',
        'J22': 'Infección aguda VRI no especificada',
        'J30': 'Rinitis alérgica',
        'J40': 'Bronquitis no especificada',
        'J45': 'Asma',
        'U07': 'COVID-19'
    }
    
    respiratory_types['Descripción'] = respiratory_types['Código'].map(code_descriptions).fillna('Otro')
    
    # Gráfico de barras de tipos de infecciones
    fig = px.bar(
        respiratory_types.head(15), 
        x='Casos', 
        y='Código',
        hover_data=['Descripción'],
        color='Casos',
        color_continuous_scale='Blues',
        orientation='h',
        title='Top 15 Tipos de Infecciones Respiratorias'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparativa de principales grupos
    st.subheader("Comparativa de Principales Grupos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorizar en grupos mayores
        df['Grupo Respiratorio'] = 'Otros'
        df.loc[df['CIE10 DP'].str.startswith(('J09', 'J10', 'J11'), na=False), 'Grupo Respiratorio'] = 'Influenza'
        df.loc[df['CIE10 DP'].str.startswith(('J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18'), na=False), 'Grupo Respiratorio'] = 'Neumonía'
        df.loc[df['CIE10 DP'].str.startswith('U07', na=False), 'Grupo Respiratorio'] = 'COVID-19'
        df.loc[df['CIE10 DP'].str.startswith(('J00', 'J01', 'J02', 'J03', 'J04', 'J05', 'J06'), na=False), 'Grupo Respiratorio'] = 'IRA Alta'
        df.loc[df['CIE10 DP'].str.startswith(('J20', 'J21', 'J22'), na=False), 'Grupo Respiratorio'] = 'IRA Baja'
        
        group_counts = df['Grupo Respiratorio'].value_counts().reset_index()
        group_counts.columns = ['Grupo', 'Casos']
        
        fig = px.pie(
            group_counts, 
            names='Grupo', 
            values='Casos',
            title='Distribución por Grupos de Enfermedades Respiratorias',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)
    
    with col2:
        # Severidad por grupo
        severity_by_group = df.groupby('Grupo Respiratorio')['Destino'].apply(
            lambda x: (x != 'DOMICILIO').mean() * 100
        ).reset_index()
        severity_by_group.columns = ['Grupo', 'Porcentaje de Derivación']
        
        fig = px.bar(
            severity_by_group,
            x='Grupo',
            y='Porcentaje de Derivación',
            title='Severidad por Grupo (% Derivación)',
            color='Porcentaje de Derivación',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig)
    
    # Evolución temporal por tipo
    st.subheader("Evolución Temporal por Tipo")
    
    # Agrupar por año, semana y grupo
    temporal_data = df.groupby(['Año', 'Semana Epidemiologica', 'Grupo Respiratorio']).size().reset_index(name='Casos')
    
    fig = px.line(
        temporal_data,
        x='Semana Epidemiologica',
        y='Casos',
        color='Grupo Respiratorio',
        facet_row='Año',
        title='Evolución Semanal por Tipo',
        category_orders={'Año': sorted(df['Año'].unique())}
    )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por edad y sexo
    st.subheader("Análisis por Edad y Sexo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por grupo de edad
        age_group_counts = df.groupby(['Grupo Respiratorio', 'Grupo_Edad']).size().reset_index(name='Casos')
        
        fig = px.bar(
            age_group_counts,
            x='Grupo_Edad',
            y='Casos',
            color='Grupo Respiratorio',
            title='Distribución por Grupo Etario',
            barmode='group'
        )
        
        st.plotly_chart(fig)
    
    with col2:
        # Distribución por sexo
        sex_counts = df.groupby(['Grupo Respiratorio', 'Sexo']).size().reset_index(name='Casos')
        
        fig = px.bar(
            sex_counts,
            x='Sexo',
            y='Casos',
            color='Grupo Respiratorio',
            title='Distribución por Sexo',
            barmode='group'
        )
        
        st.plotly_chart(fig)
    
    # Recomendaciones específicas
    st.subheader("Recomendaciones Específicas")
    
    # Determinar fase epidémica actual
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    transmissibility = enhanced_ts.get('transmissibility_metrics', {})
    current_rt = transmissibility.get('current_rt', 0)
    
    # Mensaje basado en Rt
    epidemic_phase = ""
    recommendations = []
    
    if current_rt > 1.5:
        epidemic_phase = "**Fase de crecimiento rápido**"
        recommendations = [
            "Intensificar vigilancia y seguimiento de casos",
            "Reforzar medidas preventivas en grupos de alto riesgo",
            "Preparar servicios de salud para potencial aumento de casos",
            "Considerar campañas de vacunación específicas si aplica",
            "Implementar sistemas de alerta temprana en centros educativos y laborales"
        ]
    elif current_rt > 1:
        epidemic_phase = "**Fase de crecimiento moderado**"
        recommendations = [
            "Mantener vigilancia activa de nuevos casos",
            "Reforzar medidas preventivas básicas",
            "Verificar disponibilidad de recursos en servicios de salud",
            "Identificar y proteger grupos vulnerables",
            "Monitorear complicaciones y hospitalizaciones"
        ]
    elif current_rt > 0.5:
        epidemic_phase = "**Fase de decrecimiento moderado**"
        recommendations = [
            "Mantener vigilancia básica",
            "Educar sobre medidas preventivas estándar",
            "Evaluar impacto de medidas implementadas",
            "Preparar para posible fin de temporada epidémica",
            "Planificar recursos para próxima temporada"
        ]
    else:
        epidemic_phase = "**Fase de decrecimiento rápido/control**"
        recommendations = [
            "Mantener vigilancia básica de casos",
            "Documentar lecciones aprendidas",
            "Evaluar la efectividad de intervenciones",
            "Planificar actividades de prevención fuera de temporada",
            "Optimizar recursos y preparación para próxima temporada"
        ]
    
    # Mostrar recomendaciones
    st.markdown(f"Fase epidémica actual: {epidemic_phase}")
    st.markdown("Recomendaciones:")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

def display_gastrointestinal_analysis(df, analysis_results):
    """
    Muestra el análisis específico para enfermedades gastrointestinales.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
    """
    # Análisis específico para enfermedades gastrointestinales
    gi_results = analysis_results.get('gastrointestinal_specific', {})
    
    # Top tipos de infecciones gastrointestinales
    st.subheader("Distribución de Infecciones Gastrointestinales")
    
    # Agrupar por primeros 3 caracteres de CIE10
    gi_types = df['CIE10 DP'].str[:3].value_counts().reset_index()
    gi_types.columns = ['Código', 'Casos']
    
    # Añadir descripciones
    code_descriptions = {
        'A00': 'Cólera',
        'A01': 'Fiebres tifoidea y paratifoidea',
        'A02': 'Otras infecciones por Salmonella',
        'A03': 'Shigelosis',
        'A04': 'Otras infecciones intestinales bacterianas',
        'A05': 'Intoxicación alimentaria bacteriana',
        'A06': 'Amebiasis',
        'A07': 'Otras enfermedades intestinales por protozoos',
        'A08': 'Infecciones intestinales virales',
        'A09': 'Diarrea y gastroenteritis de presunto origen infeccioso',
        'K52': 'Otras gastroenteritis y colitis no infecciosas',
        'K59': 'Otros trastornos funcionales del intestino'
    }
    
    gi_types['Descripción'] = gi_types['Código'].map(code_descriptions).fillna('Otro')
    
    # Gráfico de barras de tipos de infecciones
    fig = px.bar(
        gi_types.head(10), 
        x='Casos', 
        y='Código',
        hover_data=['Descripción'],
        color='Casos',
        color_continuous_scale='Blues',
        orientation='h',
        title='Top 10 Tipos de Infecciones Gastrointestinales'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparativa de principales grupos
    st.subheader("Comparativa de Principales Grupos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorizar en grupos mayores
        df['Grupo GI'] = 'Otros'
        df.loc[df['CIE10 DP'].str.startswith('A01', na=False), 'Grupo GI'] = 'Tifoidea/Paratifoidea'
        df.loc[df['CIE10 DP'].str.startswith('A02', na=False), 'Grupo GI'] = 'Salmonelosis'
        df.loc[df['CIE10 DP'].str.startswith('A03', na=False), 'Grupo GI'] = 'Shigelosis'
        df.loc[df['CIE10 DP'].str.startswith('A04', na=False), 'Grupo GI'] = 'Intestinal Bacteriana'
        df.loc[df['CIE10 DP'].str.startswith('A05', na=False), 'Grupo GI'] = 'Intoxicación Alimentaria'
        df.loc[df['CIE10 DP'].str.startswith(('A06', 'A07'), na=False), 'Grupo GI'] = 'Protozoarios'
        df.loc[df['CIE10 DP'].str.startswith('A08', na=False), 'Grupo GI'] = 'Viral'
        df.loc[df['CIE10 DP'].str.startswith('A09', na=False), 'Grupo GI'] = 'Diarrea/GE Infecciosa'
        df.loc[df['CIE10 DP'].str.startswith('K52', na=False), 'Grupo GI'] = 'GE No Infecciosa'
        
        group_counts = df['Grupo GI'].value_counts().reset_index()
        group_counts.columns = ['Grupo', 'Casos']
        
        fig = px.pie(
            group_counts, 
            names='Grupo', 
            values='Casos',
            title='Distribución por Grupos de Enfermedades Gastrointestinales',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)
    
    with col2:
        # Severidad por grupo
        severity_by_group = df.groupby('Grupo GI')['Destino'].apply(
            lambda x: (x != 'DOMICILIO').mean() * 100
        ).reset_index()
        severity_by_group.columns = ['Grupo', 'Porcentaje de Derivación']
        
        fig = px.bar(
            severity_by_group,
            x='Grupo',
            y='Porcentaje de Derivación',
            title='Severidad por Grupo (% Derivación)',
            color='Porcentaje de Derivación',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig)
    
    # Evolución temporal por tipo
    st.subheader("Evolución Temporal por Tipo")
    
    # Agrupar por año, semana y grupo
    temporal_data = df.groupby(['Año', 'Semana Epidemiologica', 'Grupo GI']).size().reset_index(name='Casos')
    
    fig = px.line(
        temporal_data,
        x='Semana Epidemiologica',
        y='Casos',
        color='Grupo GI',
        facet_row='Año',
        title='Evolución Semanal por Tipo',
        category_orders={'Año': sorted(df['Año'].unique())}
    )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por grupo etario
    st.subheader("Distribución por Grupo Etario")
    
    age_counts = df.groupby(['Grupo GI', 'Grupo_Edad']).size().reset_index(name='Casos')
    
    fig = px.bar(
        age_counts,
        x='Grupo_Edad',
        y='Casos',
        color='Grupo GI',
        title='Casos por Grupo Etario',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución geográfica si existe la información
    if 'Comuna' in df.columns and not df['Comuna'].isna().all():
        st.subheader("Distribución Geográfica")
        
        # Top comunas
        top_comunas = df['Comuna'].value_counts().head(10).reset_index()
        top_comunas.columns = ['Comuna', 'Casos']
        
        fig = px.bar(
            top_comunas,
            x='Comuna',
            y='Casos',
            color='Casos',
            color_continuous_scale='Blues',
            title='Top 10 Comunas'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de estacionalidad
    st.subheader("Análisis de Estacionalidad")
    
    monthly_data = df.groupby(['Año', 'Mes']).size().reset_index(name='Casos')
    
    # Añadir nombres de mes
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    monthly_data['Nombre Mes'] = monthly_data['Mes'].map(month_names)
    
    fig = px.line(
        monthly_data,
        x='Mes',
        y='Casos',
        color='Año',
        title='Casos Mensuales por Año',
        markers=True
    )
    
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=list(month_names.values()))
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones específicas
    st.subheader("Recomendaciones Específicas")
    
    # Determinar estacionalidad actual
    current_month = datetime.now().month
    season = ""
    season_recommendations = []
    
    if 11 <= current_month <= 12 or 1 <= current_month <= 3:  # Verano en hemisferio sur
        season = "**Temporada de verano**"
        season_recommendations = [
            "Vigilancia activa de brotes asociados al calor y manipulación de alimentos",
            "Campañas de educación sobre higiene alimentaria y conservación de alimentos",
            "Reforzar control sanitario en lugares de venta de alimentos",
            "Vigilancia de calidad del agua en zonas recreacionales",
            "Preparación para potencial aumento de casos en niños por actividades veraniegas"
        ]
    elif 4 <= current_month <= 5:  # Otoño
        season = "**Temporada de otoño**"
        season_recommendations = [
            "Vigilancia de transición de patógenos prevalentes",
            "Educación sobre lavado de manos en entornos escolares",
            "Monitoreo de brotes relacionados con el inicio del período escolar",
            "Preparación para temporada invernal",
            "Evaluación de patrones de resistencia antimicrobiana"
        ]
    elif 6 <= current_month <= 8:  # Invierno
        season = "**Temporada de invierno**"
        season_recommendations = [
            "Vigilancia de infecciones virales gastrointestinales",
            "Protocolos de control de infecciones en centros de salud",
            "Educación comunitaria sobre prevención de infecciones cruzadas",
            "Monitoreo de complicaciones en grupos vulnerables",
            "Optimización del diagnóstico etiológico"
        ]
    else:  # Primavera
        season = "**Temporada de primavera**"
        season_recommendations = [
            "Vigilancia de cambios en patrones epidemiológicos",
            "Educación sobre prevención durante actividades al aire libre",
            "Monitoreo de calidad del agua por lluvias estacionales",
            "Preparación para temporada estival",
            "Vigilancia de resistencia antimicrobiana"
        ]
    
    # Mostrar recomendaciones
    st.markdown(f"Estación actual: {season}")
    st.markdown("Recomendaciones estacionales:")
    
    for i, rec in enumerate(season_recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Recomendaciones generales
    st.markdown("### Recomendaciones Generales")
    
    general_recommendations = [
        "**Prevención primaria**: Higiene de manos, manipulación segura de alimentos, agua potable segura",
        "**Vigilancia**: Monitoreo de patógenos predominantes y resistencia antimicrobiana",
        "**Educación**: Difusión de medidas preventivas a población general",
        "**Atención clínica**: Uso racional de antibióticos y manejo de hidratación oral",
        "**Protección de grupos vulnerables**: Niños menores de 5 años y adultos mayores"
    ]
    
    for rec in general_recommendations:
        st.markdown(f"- {rec}")

def display_varicela_analysis(df, analysis_results):
    """
    Muestra el análisis específico para casos de varicela.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
    """
    # Análisis específico para varicela
    varicela_results = analysis_results.get('varicela_specific', {})
    
    # Distribución por edad
    st.subheader("Distribución por Edad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='Edad_Anios',
            title='Distribución por Edad',
            nbins=20,
            color_discrete_sequence=['steelblue']
        )
        
        fig.update_layout(xaxis_title='Edad (años)', yaxis_title='Número de Casos')
        st.plotly_chart(fig)
    
    with col2:
        # Estadísticas descriptivas
        age_stats = df['Edad_Anios'].describe().to_dict()
        
        st.markdown("### Estadísticas de Edad")
        st.markdown(f"- **Media**: {age_stats['mean']:.2f} años")
        st.markdown(f"- **Mediana**: {age_stats['50%']:.2f} años")
        st.markdown(f"- **Desviación estándar**: {age_stats['std']:.2f}")
        st.markdown(f"- **Mínimo**: {age_stats['min']:.2f} años")
        st.markdown(f"- **Máximo**: {age_stats['max']:.2f} años")
        
        # Proporción en niños
        children = len(df[df['Edad_Anios'] < 15])
        total = len(df)
        st.markdown(f"- **Proporción en menores de 15 años**: {children/total*100:.1f}%")
    
    # Distribución temporal
    st.subheader("Distribución Temporal")
    
    # Datos mensuales
    monthly_data = df.groupby(['Año', 'Mes']).size().reset_index(name='Casos')
    
    # Añadir nombres de mes
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    monthly_data['Nombre Mes'] = monthly_data['Mes'].map(month_names)
    
    fig = px.line(
        monthly_data,
        x='Mes',
        y='Casos',
        color='Año',
        title='Casos Mensuales por Año',
        markers=True
    )
    
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=list(month_names.values()))
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap por año y mes
    monthly_pivot = monthly_data.pivot_table(
        values='Casos',
        index='Mes',
        columns='Año',
        fill_value=0
    )
    
    fig = px.imshow(
        monthly_pivot,
        labels=dict(x="Año", y="Mes", color="Casos"),
        y=[month_names[m] for m in sorted(monthly_pivot.index)],
        x=sorted(monthly_pivot.columns),
        color_continuous_scale='Blues',
        title='Mapa de Calor: Casos por Mes y Año'
    )
    
    fig.update_layout(coloraxis_showscale=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de severidad
    st.subheader("Análisis de Severidad")
    
    destinos = df['Destino'].value_counts(normalize=True).reset_index()
    destinos.columns = ['Destino', 'Proporción']
    destinos['Proporción'] = destinos['Proporción'] * 100
    
    fig = px.pie(
        destinos,
        names='Destino',
        values='Proporción',
        title='Distribución por Destino',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Severidad por grupo de edad
    severity_by_age = df.groupby('Grupo_Edad')['Destino'].apply(
        lambda x: (x != 'DOMICILIO').mean() * 100
    ).reset_index()
    severity_by_age.columns = ['Grupo de Edad', 'Porcentaje de Derivación']
    
    fig = px.bar(
        severity_by_age,
        x='Grupo de Edad',
        y='Porcentaje de Derivación',
        title='Severidad por Grupo Etario (% Derivación)',
        color='Porcentaje de Derivación',
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones específicas
    st.subheader("Recomendaciones Específicas")
    
    # Análisis de ciclos y estacionalidad
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    cycles = enhanced_ts.get('epidemic_cycles', {})
    
    # Determinar fase epidémica
    if cycles.get('cycle_detected', False):
        if 'next_outbreak_estimate' in cycles:
            next_outbreak = cycles['next_outbreak_estimate']
            if hasattr(next_outbreak, 'strftime'):
                today = datetime.now()
                days_to_outbreak = (next_outbreak - today).days
                
                if days_to_outbreak < 0:
                    phase = "**Posible fase epidémica activa**"
                    recommendations = [
                        "Intensificar vigilancia activa de casos en centros educativos",
                        "Reforzar medidas de aislamiento de casos",
                        "Vigilar complicaciones en grupos de riesgo (embarazadas, inmunocomprometidos)",
                        "Verificar disponibilidad de antivirales para casos graves/complicados",
                        "Educación sobre prevención de cicatrices y sobreinfección"
                    ]
                elif days_to_outbreak < 90:
                    phase = f"**Fase pre-epidémica** (próximo brote estimado en {days_to_outbreak} días)"
                    recommendations = [
                        "Preparar sistemas de vigilancia para detección temprana",
                        "Revisar protocolos de manejo y notificación de casos",
                        "Educación anticipada a comunidades educativas",
                        "Verificar cobertura de vacunación en población objetivo",
                        "Preparar insumos y recursos para manejo de casos"
                    ]
                else:
                    phase = "**Fase inter-epidémica**"
                    recommendations = [
                        "Mantener vigilancia pasiva de casos",
                        "Fomentar vacunación en población susceptible",
                        "Educación sobre reconocimiento temprano de síntomas",
                        "Actualización de guías clínicas y protocolos",
                        "Análisis retrospectivo de últimos brotes"
                    ]
            else:
                phase = "**Fase indeterminada**"
                recommendations = [
                    "Mantener vigilancia habitual",
                    "Fomentar vacunación en población susceptible",
                    "Educación sobre prevención y control",
                    "Preparación general de servicios de salud",
                    "Monitoreo de casos esporádicos"
                ]
        else:
            phase = "**Fase indeterminada**"
            recommendations = [
                "Mantener vigilancia habitual",
                "Fomentar vacunación en población susceptible",
                "Educación sobre prevención y control",
                "Preparación general de servicios de salud",
                "Monitoreo de casos esporádicos"
            ]
    else:
        phase = "**Fase indeterminada (no se detectó ciclo)**"
        recommendations = [
            "Mantener vigilancia habitual",
            "Fomentar vacunación en población susceptible",
            "Educación sobre prevención y control",
            "Preparación general de servicios de salud",
            "Monitoreo de casos esporádicos"
        ]
    
    # Mostrar fase y recomendaciones
    st.markdown(f"Fase epidémica actual: {phase}")
    st.markdown("Recomendaciones:")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Información sobre vacunación
    st.markdown("### Prevención mediante Vacunación")
    
    vaccine_info = """
    La varicela es una enfermedad prevenible por vacunación. En Chile, la vacuna contra la varicela:
    
    - Está incluida en el Programa Nacional de Inmunizaciones desde 2014
    - Se administra a los 18 meses de edad (1 dosis)
    - Indicaciones especiales en brotes y para grupos de riesgo
    - Efectividad: 80-85% para prevenir cualquier forma de enfermedad, >95% para prevenir formas graves
    
    La inmunidad de rebaño se alcanza con coberturas superiores al 85-90%.
    """
    
    st.markdown(vaccine_info)

def display_manopieboca_analysis(df, analysis_results):
    """
    Muestra el análisis específico para casos de enfermedad mano-pie-boca.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis avanzado
    """
    # Análisis específico para enfermedad mano-pie-boca
    mpb_results = analysis_results.get('manopieboca_specific', {})
    
    # Distribución por edad con énfasis en menores de 10 años
    st.subheader("Distribución por Edad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtrar para mostrar principalmente menores de 10 años
        df_mpb_age = df[df['Edad_Anios'] <= 10].copy()
        
        fig = px.histogram(
            df_mpb_age,
            x='Edad_Anios',
            title='Distribución por Edad (0-10 años)',
            nbins=11,  # Un bin por año de 0 a 10
            color_discrete_sequence=['steelblue']
        )
        
        fig.update_layout(xaxis_title='Edad (años)', yaxis_title='Número de Casos')
        st.plotly_chart(fig)
    
    with col2:
        # Estadísticas descriptivas de edad
        age_stats = df['Edad_Anios'].describe().to_dict()
        
        st.markdown("### Estadísticas de Edad")
        st.markdown(f"- **Media**: {age_stats['mean']:.2f} años")
        st.markdown(f"- **Mediana**: {age_stats['50%']:.2f} años")
        st.markdown(f"- **Desviación estándar**: {age_stats['std']:.2f}")
        st.markdown(f"- **Mínimo**: {age_stats['min']:.2f} años")
        st.markdown(f"- **Máximo**: {age_stats['max']:.2f} años")
        
        # Proporción en niños pequeños
        under_5 = len(df[df['Edad_Anios'] < 5])
        total = len(df)
        st.markdown(f"- **Proporción en menores de 5 años**: {under_5/total*100:.1f}%")
    
    # Distribución temporal
    st.subheader("Distribución Temporal")
    
    # Datos mensuales
    monthly_data = df.groupby(['Año', 'Mes']).size().reset_index(name='Casos')
    
    # Añadir nombres de mes
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    monthly_data['Nombre Mes'] = monthly_data['Mes'].map(month_names)
    
    fig = px.line(
        monthly_data,
        x='Mes',
        y='Casos',
        color='Año',
        title='Casos Mensuales por Año',
        markers=True
    )
    
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=list(month_names.values()))
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap por año y mes
    monthly_pivot = monthly_data.pivot_table(
        values='Casos',
        index='Mes',
        columns='Año',
        fill_value=0
    )
    
    fig = px.imshow(
        monthly_pivot,
        labels=dict(x="Año", y="Mes", color="Casos"),
        y=[month_names[m] for m in sorted(monthly_pivot.index)],
        x=sorted(monthly_pivot.columns),
        color_continuous_scale='Blues',
        title='Mapa de Calor: Casos por Mes y Año'
    )
    
    fig.update_layout(coloraxis_showscale=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de severidad
    st.subheader("Análisis de Severidad")
    
    destinos = df['Destino'].value_counts(normalize=True).reset_index()
    destinos.columns = ['Destino', 'Proporción']
    destinos['Proporción'] = destinos['Proporción'] * 100
    
    fig = px.pie(
        destinos,
        names='Destino',
        values='Proporción',
        title='Distribución por Destino',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución geográfica si existe la información
    if 'Comuna' in df.columns and not df['Comuna'].isna().all():
        st.subheader("Distribución Geográfica")
        
        # Top comunas
        top_comunas = df['Comuna'].value_counts().head(10).reset_index()
        top_comunas.columns = ['Comuna', 'Casos']
        
        fig = px.bar(
            top_comunas,
            x='Comuna',
            y='Casos',
            color='Casos',
            color_continuous_scale='Blues',
            title='Top 10 Comunas'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones específicas
    st.subheader("Recomendaciones Específicas")
    
    # Análisis de ciclos y estacionalidad
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    cycles = enhanced_ts.get('epidemic_cycles', {})
    
    # Determinar estacionalidad actual
    current_month = datetime.now().month
    
    if 9 <= current_month <= 12 or 1 <= current_month <= 3:  # Primavera-Verano en hemisferio sur
        season = "**Alta temporada (Primavera-Verano)**"
        season_recommendations = [
            "Educación a jardines infantiles y colegios sobre prevención y control",
            "Capacitación a personal de salud para diagnóstico oportuno",
            "Vigilancia activa en centros educativos y de cuidado infantil",
            "Difusión de medidas preventivas (lavado de manos, limpieza de superficies)",
            "Medidas de control de casos en centros educativos (exclusión hasta mejoría de lesiones)"
        ]
    else:  # Otoño-Invierno
        season = "**Baja temporada (Otoño-Invierno)**"
        season_recommendations = [
            "Mantener vigilancia pasiva de casos",
            "Educación continua sobre higiene de manos en centros infantiles",
            "Preparación para temporada de primavera-verano",
            "Análisis retrospectivo de casos de la temporada anterior",
            "Actualización de protocolos y guías de manejo clínico"
        ]
    
    # Determinar fase de ciclo epidémico
    if cycles.get('cycle_detected', False):
        if 'next_outbreak_estimate' in cycles:
            next_outbreak = cycles['next_outbreak_estimate']
            if hasattr(next_outbreak, 'strftime'):
                today = datetime.now()
                days_to_outbreak = (next_outbreak - today).days
                
                if days_to_outbreak < 0:
                    cycle_phase = "**Posible fase epidémica activa**"
                elif days_to_outbreak < 60:
                    cycle_phase = f"**Fase pre-epidémica** (próximo brote estimado en {days_to_outbreak} días)"
                else:
                    cycle_phase = "**Fase inter-epidémica**"
            else:
                cycle_phase = "**Fase indeterminada**"
        else:
            cycle_phase = "**Fase indeterminada**"
    else:
        cycle_phase = "**Fase indeterminada (no se detectó ciclo)**"
    
    # Mostrar estacionalidad y fase del ciclo
    st.markdown(f"Estacionalidad actual: {season}")
    st.markdown(f"Fase del ciclo epidémico: {cycle_phase}")
    
    st.markdown("**Recomendaciones estacionales:**")
    for i, rec in enumerate(season_recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Información sobre prevención
    st.markdown("### Prevención y Control")
    
    prevention_info = """
    La enfermedad mano-pie-boca es causada principalmente por el virus Coxsackie A16 y el Enterovirus 71.
    
    **Medidas preventivas clave:**
    
    - Lavado frecuente de manos, especialmente después de cambiar pañales o ir al baño
    - Desinfección de superficies y juguetes compartidos
    - Evitar contacto cercano con personas infectadas
    - Exclusión temporal de niños con lesiones activas de centros educativos
    - Mantener buena ventilación en espacios compartidos
    
    **Manejo de brotes en establecimientos educacionales:**
    
    1. Notificación oportuna a autoridades sanitarias
    2. Refuerzo de medidas de higiene (especialmente lavado de manos)
    3. Limpieza profunda de áreas comunes
    4. Educación a padres sobre reconocimiento de síntomas
    5. Exclusión de casos hasta la resolución de síntomas (generalmente 7-10 días)
    """
    
    st.markdown(prevention_info)

def display_gpt_analysis(analysis_results, analysis_type):
    """
    Muestra el análisis generado por GPT.
    
    Args:
        analysis_results (dict): Resultados del análisis avanzado
        analysis_type (str): Tipo de análisis
    """
    st.header("📝 Análisis GPT: Interpretación Epidemiológica")
    
    # Verificar si hay análisis GPT disponible
    gpt_file_path = f'analisis/gpt_analysis_{analysis_type}.txt'
    
    if os.path.exists(gpt_file_path):
        with open(gpt_file_path, 'r', encoding='utf-8') as f:
            gpt_analysis = f.read()
        
        # Opción para generar un nuevo análisis
        if st.button("Generar nuevo análisis con GPT"):
            st.info("Esta funcionalidad requeriría conexión con la API de GPT. El nuevo análisis reemplazaría el existente.")
        
        # Mostrar el análisis GPT
        st.markdown(gpt_analysis)
        
        # Opciones para exportar
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Exportar como PDF"):
                st.info("Esta funcionalidad generaría un PDF del análisis.")
        
        with export_col2:
            if st.button("Compartir análisis"):
                st.info("Esta funcionalidad permitiría compartir el análisis por correo electrónico o enlace.")
    else:
        st.warning(f"No se encontró un análisis GPT para {analysis_type}. Genere uno nuevo con el botón a continuación.")
        
        if st.button("Generar análisis con GPT"):
            with st.spinner("Generando análisis con GPT... Esta operación puede tomar unos momentos..."):
                st.info("En una implementación completa, aquí se llamaría a la API de GPT para generar un nuevo análisis.")
                
                # Generación simulada
                st.success("Análisis generado correctamente.")
        
        # Mostrar análisis de ejemplo
        st.markdown("""
        ## Análisis Epidemiológico de Ejemplo
        
        ### Resumen Ejecutivo
        
        El análisis de datos epidemiológicos para [TIPO ANÁLISIS] en San Pedro de la Paz revela un patrón estacional 
        con picos en los meses de invierno y una tendencia general [CRECIENTE/DECRECIENTE] en los últimos años. 
        
        ### Interpretación detallada
        
        Los datos sugieren una periodicidad clara con ciclos que pueden relacionarse con factores climáticos y sociales...
        
        ### Recomendaciones
        
        1. Reforzar la vigilancia epidemiológica durante los meses previos al pico estacional
        2. Implementar estrategias preventivas focalizadas en grupos de mayor riesgo
        3. ...
        
        *Nota: Este es un análisis de ejemplo. Genere un análisis real con el botón superior.*
        """)
    
    # Información sobre metodología
    with st.expander("Metodología del Análisis GPT"):
        st.markdown("""
        ### Metodología del Análisis GPT
        
        El análisis generado por GPT se basa en una evaluación integral de los resultados del análisis estadístico avanzado, incluidos:
        
        - Análisis de tendencias temporales y estacionalidad
        - Identificación de patrones cíclicos y brotes
        - Detección de cambios estructurales en la serie temporal
        - Evaluación de métricas de transmisibilidad (Rt)
        - Modelos predictivos ARIMA/SARIMA
        - Análisis demográfico y de severidad
        
        El modelo interpreta estos datos en el contexto específico de la epidemiología de la enfermedad, 
        considerando factores como la estacionalidad conocida, los mecanismos de transmisión, 
        la población afectada y las medidas de control disponibles.
        
        **Limitaciones:**
        - El análisis se basa únicamente en los datos proporcionados
        - No incluye información sobre intervenciones específicas implementadas
        - No considera factores sociales, económicos o ambientales no incluidos en los datos
        """)

def analyze_age_groups(df):
    """
    Analiza los datos por grupo de edad.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        
    Returns:
        dict: Resultados del análisis por grupo de edad
    """
    results = {}
    
    # Conteo por grupo de edad
    age_counts = df['Grupo_Edad'].value_counts().reset_index()
    age_counts.columns = ['Grupo de Edad', 'Casos']
    results['counts'] = age_counts
    
    # Estadísticas por grupo de edad
    age_stats = df.groupby('Grupo_Edad')['Edad_Anios'].agg(['mean', 'median', 'std']).reset_index()
    results['stats'] = age_stats
    
    # Severidad por grupo de edad
    severity_by_age = df.groupby('Grupo_Edad')['Destino'].apply(
        lambda x: (x != 'DOMICILIO').mean() * 100
    ).reset_index()
    severity_by_age.columns = ['Grupo de Edad', 'Porcentaje de Derivación']
    results['severity'] = severity_by_age
    
    return results

def display_age_group_analysis(age_analysis):
    """
    Muestra el análisis por grupo de edad.
    
    Args:
        age_analysis (dict): Resultados del análisis por grupo de edad
    """
    if 'counts' in age_analysis:
        # Gráfico de distribución
        fig = px.bar(
            age_analysis['counts'],
            x='Grupo de Edad',
            y='Casos',
            title='Distribución de Casos por Grupo de Edad',
            color='Casos',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if 'severity' in age_analysis:
        # Gráfico de severidad
        fig = px.bar(
            age_analysis['severity'],
            x='Grupo de Edad',
            y='Porcentaje de Derivación',
            title='Severidad por Grupo de Edad (% Derivación)',
            color='Porcentaje de Derivación',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if 'stats' in age_analysis:
        # Tabla de estadísticas
        st.markdown("### Estadísticas por Grupo de Edad")
        
        stats_df = age_analysis['stats'].copy()
        stats_df.columns = ['Grupo de Edad', 'Media', 'Mediana', 'Desv. Estándar']
        stats_df = stats_df.round(2)
        
        st.dataframe(stats_df)

def analyze_gender(df):
    """
    Analiza los datos por género.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        
    Returns:
        dict: Resultados del análisis por género
    """
    results = {}
    
    # Conteo por género
    gender_counts = df['Sexo'].value_counts().reset_index()
    gender_counts.columns = ['Género', 'Casos']
    results['counts'] = gender_counts
    
    # Estadísticas por género
    gender_stats = df.groupby('Sexo')['Edad_Anios'].agg(['mean', 'median', 'std']).reset_index()
    results['stats'] = gender_stats
    
    # Severidad por género
    severity_by_gender = df.groupby('Sexo')['Destino'].apply(
        lambda x: (x != 'DOMICILIO').mean() * 100
    ).reset_index()
    severity_by_gender.columns = ['Género', 'Porcentaje de Derivación']
    results['severity'] = severity_by_gender
    
    # Distribución de enfermedades por género (si aplica)
    if 'Grupo Respiratorio' in df.columns:
        disease_by_gender = df.groupby(['Sexo', 'Grupo Respiratorio']).size().reset_index(name='Casos')
        results['diseases'] = disease_by_gender
    elif 'Grupo GI' in df.columns:
        disease_by_gender = df.groupby(['Sexo', 'Grupo GI']).size().reset_index(name='Casos')
        results['diseases'] = disease_by_gender
    
    return results

def display_gender_analysis(gender_analysis):
    """
    Muestra el análisis por género.
    
    Args:
        gender_analysis (dict): Resultados del análisis por género
    """
    if 'counts' in gender_analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribución
            fig = px.pie(
                gender_analysis['counts'],
                names='Género',
                values='Casos',
                title='Distribución por Género',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)
        
        with col2:
            if 'severity' in gender_analysis:
                # Gráfico de severidad
                fig = px.bar(
                    gender_analysis['severity'],
                    x='Género',
                    y='Porcentaje de Derivación',
                    title='Severidad por Género (% Derivación)',
                    color='Porcentaje de Derivación',
                    color_continuous_scale='Reds'
                )
                
                st.plotly_chart(fig)
    
    if 'stats' in gender_analysis:
        # Tabla de estadísticas
        st.markdown("### Estadísticas por Género")
        
        stats_df = gender_analysis['stats'].copy()
        stats_df.columns = ['Género', 'Edad Media', 'Edad Mediana', 'Desv. Estándar']
        stats_df = stats_df.round(2)
        
        st.dataframe(stats_df)
    
    if 'diseases' in gender_analysis:
        # Gráfico de distribución de enfermedades por género
        st.markdown("### Distribución de Enfermedades por Género")
        
        fig = px.bar(
            gender_analysis['diseases'],
            x='Género',
            y='Casos',
            color=gender_analysis['diseases'].columns[1],  # 'Grupo Respiratorio' o 'Grupo GI'
            title='Distribución de Enfermedades por Género',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def analyze_establishments(df):
    """
    Analiza los datos por establecimiento.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        
    Returns:
        dict: Resultados del análisis por establecimiento
    """
    if 'Estableciemiento' not in df.columns or df['Estableciemiento'].isna().all():
        return None
    
    results = {}
    
    # Conteo por establecimiento
    estab_counts = df['Estableciemiento'].value_counts().reset_index()
    estab_counts.columns = ['Estableciemiento', 'Casos']
    results['counts'] = estab_counts
    
    # Estadísticas por establecimiento
    estab_stats = df.groupby('Estableciemiento')['Edad_Anios'].agg(['mean', 'median', 'std']).reset_index()
    results['stats'] = estab_stats
    
    # Severidad por establecimiento
    severity_by_estab = df.groupby('Estableciemiento')['Destino'].apply(
        lambda x: (x != 'DOMICILIO').mean() * 100
    ).reset_index()
    severity_by_estab.columns = ['Estableciemiento', 'Porcentaje de Derivación']
    results['severity'] = severity_by_estab
    
    # Casos por año y establecimiento
    yearly_estab = df.groupby(['Año', 'Estableciemiento']).size().reset_index(name='Casos')
    results['yearly'] = yearly_estab
    
    return results

def display_establishment_analysis(establishment_analysis):
    """
    Muestra el análisis por establecimiento.
    
    Args:
        establishment_analysis (dict): Resultados del análisis por establecimiento
    """
    if establishment_analysis is None:
        st.warning("No hay datos de establecimiento disponibles.")
        return
    
    if 'counts' in establishment_analysis:
        # Gráfico de distribución
        fig = px.bar(
            establishment_analysis['counts'],
            x='Estableciemiento',
            y='Casos',
            title='Distribución de Casos por Estableciemiento',
            color='Casos',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if 'yearly' in establishment_analysis:
        # Gráfico de tendencia por establecimiento
        fig = px.line(
            establishment_analysis['yearly'],
            x='Año',
            y='Casos',
            color='Estableciemiento',
            title='Tendencia Anual por Estableciemiento',
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if 'severity' in establishment_analysis:
        # Gráfico de severidad
        fig = px.bar(
            establishment_analysis['severity'],
            x='Estableciemiento',
            y='Porcentaje de Derivación',
            title='Severidad por Estableciemiento (% Derivación)',
            color='Porcentaje de Derivación',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if 'stats' in establishment_analysis:
        # Tabla de estadísticas
        st.markdown("### Estadísticas por Establecimiento")
        
        stats_df = establishment_analysis['stats'].copy()
        stats_df.columns = ['Estableciemiento', 'Edad Media', 'Edad Mediana', 'Desv. Estándar']
        stats_df = stats_df.round(2)
        
        st.dataframe(stats_df)

def display_raw_data(df):
    """
    Muestra los datos crudos en formato tabular.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
    """
    st.header("📋 Datos")
    
    # Opciones de visualización
    st.subheader("Opciones de Visualización")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_rows = st.slider("Número de filas a mostrar", 5, 100, 10)
    
    with col2:
        show_stats = st.checkbox("Mostrar estadísticas descriptivas", True)
    
    # Mostrar datos
    st.subheader("Vista de Datos")
    
    st.dataframe(df.head(n_rows))
    
    # Estadísticas descriptivas
    if show_stats:
        st.subheader("Estadísticas Descriptivas")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        else:
            st.info("No hay columnas numéricas para calcular estadísticas.")
    
# Información del DataFrame
    st.subheader("Información del DataFrame")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Número de filas**: {df.shape[0]}")
        st.markdown(f"**Número de columnas**: {df.shape[1]}")
    
    with col2:
        st.markdown(f"**Rango de fechas**: {df['Fecha Admision'].min().strftime('%d/%m/%Y')} - {df['Fecha Admision'].max().strftime('%d/%m/%Y')}")
        st.markdown(f"**Años incluidos**: {', '.join(map(str, sorted(df['Año'].unique())))}")
    
    # Valores faltantes
    st.subheader("Valores Faltantes")
    
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores Faltantes': missing_data.values,
            'Porcentaje': (missing_data.values / len(df) * 100).round(2)
        })
        
        st.dataframe(missing_df)
    else:
        st.markdown("No hay valores faltantes en el conjunto de datos filtrado.")
    
    # Acciones adicionales
    st.subheader("Acciones Adicionales")
    
    if st.button("Descargar Datos Filtrados (CSV)"):
        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="datos_filtrados.csv">Descargar CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

def export_data(df, analysis_results, analysis_type, export_format):
    """
    Exporta los datos y resultados del análisis.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_results (dict): Resultados del análisis
        analysis_type (str): Tipo de análisis
        export_format (str): Formato de exportación ('CSV', 'Excel', 'JSON', 'PDF')
    """
    # Implementación básica, en producción tendría más funcionalidades
    if export_format == 'CSV':
        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="datos_{analysis_type}.csv">Descargar CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.success("Datos exportados en formato CSV.")
    
    elif export_format == 'Excel':
        st.sidebar.info("Exportación a Excel no implementada en esta versión.")
    
    elif export_format == 'JSON':
        st.sidebar.info("Exportación a JSON no implementada en esta versión.")
    
    elif export_format == 'PDF':
        st.sidebar.info("Exportación a PDF no implementada en esta versión.")

if __name__ == '__main__':
    main()