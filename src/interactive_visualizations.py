import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

def get_year_max_weeks(year):
    """
    Obtiene el número máximo de semanas para un año específico.
    """
    return 53 if year == 2025 else 52

def prepare_data(df):
    """
    Prepara los datos asegurando el manejo correcto de las semanas epidemiológicas.
    
    Args:
        df (pandas.DataFrame): DataFrame original
        
    Returns:
        pandas.DataFrame: DataFrame procesado
    """
    if 'Casos' not in df.columns:
        weekly_data = []
        for year in df['Año'].unique():
            df_year = df[df['Año'] == year]
            max_weeks = get_year_max_weeks(year)
            
            for week in range(1, max_weeks + 1):
                cases = df_year[df_year['Semana Epidemiologica'] == week]['CIE10 DP'].count()
                weekly_data.append({
                    'Año': year,
                    'Semana Epidemiologica': week,
                    'Casos': cases
                })
        
        df = pd.DataFrame(weekly_data)
    
    return df

def plot_interactive_weekly_cases_by_age_group(df):
    """
    Crea una visualización interactiva de casos semanales por grupo de edad.
    """
    df_weekly = []
    age_groups = ['Menor de 1 A', '1 a 4 A', '5 a 14 A', '15 a 64 A', '65 y más A']
    
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)
        
        for week in range(1, max_weeks + 1):
            week_data = df_year[df_year['Semana Epidemiologica'] == week]
            week_counts = week_data['Grupo_Edad'].value_counts()
            
            for grupo in age_groups:
                data_row = {
                    'Año': year,
                    'Semana Epidemiologica': week,
                    'Grupo_Edad': grupo,
                    'Casos': week_counts.get(grupo, 0)
                }
                df_weekly.append(data_row)
    
    df_weekly = pd.DataFrame(df_weekly)
    
    fig = px.bar(df_weekly, 
                 x='Semana Epidemiologica', 
                 y='Casos',
                 color='Grupo_Edad',
                 facet_row='Año',
                 height=800,
                 title='Atenciones Semanales por Rango Etario y Año')
    
    fig.update_layout(
        xaxis_title='Semana Epidemiológica',
        yaxis_title='Número de Atenciones',
        barmode='stack',
        showlegend=True,
        legend_title='Grupo Etario',
        hovermode='x unified'
    )
    
    return fig

def plot_interactive_weekly_cases(df):
    """
    Crea una visualización interactiva de casos semanales.
    """
    df = prepare_data(df)
    
    fig = px.line(df, 
                  x='Semana Epidemiologica',
                  y='Casos',
                  color='Año',
                  title='Casos Semanales por Año',
                  markers=True)
    
    fig.update_layout(
        xaxis_title='Semana Epidemiológica',
        yaxis_title='Número de Casos',
        hovermode='x unified',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    
    # Añadir líneas verticales para los trimestres
    for quarter in [13, 26, 39]:
        fig.add_vline(x=quarter, 
                     line_dash="dash", 
                     line_color="gray",
                     opacity=0.5)
    
    return fig

def plot_interactive_age_distribution(df):
    """
    Crea una visualización interactiva de la distribución por edad.
    """
    fig = px.histogram(df,
                      x='Edad',
                      color='Sexo',
                      nbins=50,
                      title='Distribución de Edad por Sexo',
                      marginal='box')
    
    fig.update_layout(
        xaxis_title='Edad',
        yaxis_title='Frecuencia',
        bargap=0.1,
        hovermode='x unified'
    )
    
    return fig

def plot_interactive_heatmap(df):
    """
    Crea un heatmap interactivo de casos.
    """
    df = prepare_data(df)
    df['Mes'] = df['Semana Epidemiologica'].apply(lambda x: ((x-1) // 4) + 1)
    
    heatmap_data = df.pivot_table(
        values='Casos',
        index='Mes',
        columns='Año',
        aggfunc='sum'
    )
    
    fig = px.imshow(heatmap_data,
                    title='Mapa de Calor: Casos por Mes y Año',
                    aspect='auto',
                    color_continuous_scale='YlOrRd')
    
    fig.update_layout(
        xaxis_title='Año',
        yaxis_title='Mes',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear')
    )
    
    return fig

def plot_diagnoses_treemap(df):
    """
    Crea un treemap interactivo de diagnósticos.
    """
    diagnoses = df.groupby(['Año', 'Diagnostico Principal'])['CIE10 DP'].count().reset_index()
    diagnoses.columns = ['Año', 'Diagnóstico', 'Casos']
    
    fig = px.treemap(diagnoses,
                     path=['Año', 'Diagnóstico'],
                     values='Casos',
                     title='Distribución de Diagnósticos por Año')
    
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Casos: %{value}<extra></extra>')
    
    return fig

def plot_age_group_comparison(df):
    """
    Crea una visualización interactiva de comparación por grupos de edad.
    """
    fig = px.box(df,
                 x='Grupo_Edad',
                 y='Edad',
                 color='Año',
                 title='Comparación de Edades por Grupo',
                 points='all')
    
    fig.update_layout(
        xaxis_title='Grupo de Edad',
        yaxis_title='Edad',
        boxmode='group',
        showlegend=True
    )
    
    return fig

def plot_weekly_trend(df):
    """
    Crea una visualización interactiva de tendencia semanal.
    """
    df = prepare_data(df)
    
    fig = px.line(df,
                  x='Semana Epidemiologica',
                  y='Casos',
                  color='Año',
                  title='Tendencia Semanal de Casos por Año',
                  line_shape='spline',
                  markers=True)
    
    fig.update_layout(
        xaxis_title='Semana Epidemiológica',
        yaxis_title='Número de Casos',
        hovermode='x unified'
    )
    
    return fig

# ----- NUEVAS FUNCIONES PARA VISUALIZACIONES INTERACTIVAS AVANZADAS -----

def plot_interactive_structural_changes(df_weekly, enhanced_ts_results):
    """
    Crea una visualización interactiva de cambios estructurales en la serie temporal.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    if not isinstance(df_weekly.index, pd.DatetimeIndex):
        # Si el índice no es de tipo fecha, intentar convertirlo
        df_weekly = df_weekly.copy()
        if 'fecha' in df_weekly.columns:
            df_weekly.set_index('fecha', inplace=True)
        else:
            # No podemos proceder sin fechas
            fig = go.Figure()
            fig.add_annotation(
                text="No se pueden visualizar cambios estructurales sin información de fechas",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Análisis de Cambios Estructurales - No disponible")
            return fig
    
    structural_changes = enhanced_ts_results.get('structural_changes', {})
    change_points = structural_changes.get('change_points', [])
    
    # Crear gráfico base con datos originales
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_weekly.index,
        y=df_weekly['Casos'],
        mode='lines+markers',
        name='Casos semanales',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color='blue'),
        hovertemplate='%{x}<br>Casos: %{y}<extra></extra>'
    ))
    
    # Añadir líneas verticales para los cambios estructurales
    if change_points:
        for cp in change_points:
            if cp < len(df_weekly):
                date = df_weekly.index[cp]
                cases = df_weekly['Casos'].iloc[cp]
                
                # Añadir línea vertical
                fig.add_shape(
                    type="line",
                    x0=date, y0=0,
                    x1=date, y1=df_weekly['Casos'].max() * 1.1,
                    line=dict(color="red", width=2, dash="dash"),
                )
                
                # Añadir punto de cambio
                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[cases],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name=f'Cambio estructural',
                    hovertemplate='Cambio estructural<br>%{x}<br>Casos: %{y}<extra></extra>'
                ))
    
    # Añadir información sobre segmentos
    if 'segment_stats' in structural_changes:
        for stat in structural_changes['segment_stats']:
            if 'change_point' in stat and stat['change_point'] < len(df_weekly):
                date = df_weekly.index[stat['change_point']]
                rel_change = stat.get('relative_change', 0)
                pre_mean = stat.get('pre_mean', 0)
                post_mean = stat.get('post_mean', 0)
                p_value = stat.get('p_value', 1)
                
                # Información para el hover
                hover_text = (f"Cambio: {rel_change:.1%}<br>"
                            f"Media anterior: {pre_mean:.1f}<br>"
                            f"Media posterior: {post_mean:.1f}<br>"
                            f"p-valor: {p_value:.4f}<br>"
                            f"Significativo: {'Sí' if p_value < 0.05 else 'No'}")
                
                # Añadir anotación
                fig.add_annotation(
                    x=date,
                    y=df_weekly['Casos'].max() * 0.9,
                    text=f"{'+' if rel_change > 0 else ''}{rel_change:.1%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='red',
                    arrowsize=1,
                    arrowwidth=2,
                    ax=0,
                    ay=-40,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='red',
                    borderwidth=2,
                    hovertext=hover_text
                )
    
    # Configurar diseño
    fig.update_layout(
        title='Análisis de Cambios Estructurales en la Serie Temporal',
        xaxis_title='Fecha',
        yaxis_title='Número de Casos',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_interactive_arima_forecast(df_weekly, enhanced_ts_results):
    """
    Crea una visualización interactiva del pronóstico ARIMA.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    if not isinstance(df_weekly.index, pd.DatetimeIndex):
        # Si el índice no es de tipo fecha, intentar convertirlo
        df_weekly = df_weekly.copy()
        if 'fecha' in df_weekly.columns:
            df_weekly.set_index('fecha', inplace=True)
        else:
            # No podemos proceder sin fechas
            fig = go.Figure()
            fig.add_annotation(
                text="No se pueden visualizar pronósticos sin información de fechas",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Pronóstico ARIMA - No disponible")
            return fig
    
    forecast = enhanced_ts_results.get('forecast')
    
    if forecast is None or not isinstance(forecast, pd.DataFrame) or forecast.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Pronóstico no disponible",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Pronóstico ARIMA - No disponible")
        return fig
    
    # Crear gráfico base con datos históricos
    fig = go.Figure()
    
    # Añadir datos históricos
    fig.add_trace(go.Scatter(
        x=df_weekly.index,
        y=df_weekly['Casos'],
        mode='lines+markers',
        name='Datos históricos',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color='blue'),
        hovertemplate='%{x}<br>Casos: %{y}<extra></extra>'
    ))
    
    # Añadir pronóstico
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['forecast'],
        mode='lines+markers',
        name='Pronóstico',
        line=dict(color='red', width=2),
        marker=dict(size=6, color='red'),
        hovertemplate='%{x}<br>Pronóstico: %{y:.1f}<extra></extra>'
    ))
    
    # Añadir intervalo de confianza
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=forecast['upper_ci'].tolist() + forecast['lower_ci'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        hoverinfo='skip',
        showlegend=True,
        name='Intervalo de confianza 95%'
    ))
    
    # Añadir información del modelo
    model_info = enhanced_ts_results.get('model_info', {})
    model_fit = enhanced_ts_results.get('model_fit', {})
    
    if model_info:
        order = model_info.get('order', 'N/A')
        seasonal_order = model_info.get('seasonal_order', 'N/A')
        
        model_text = f"<b>Modelo ARIMA{order}"
        if seasonal_order != 'N/A':
            model_text += f" x {seasonal_order}</b>"
        else:
            model_text += "</b>"
        
        if model_fit:
            rmse = model_fit.get('rmse', 'N/A')
            if isinstance(rmse, (int, float)):
                model_text += f"<br>RMSE: {rmse:.2f}"
            
            mape = model_fit.get('mape', 'N/A')
            if isinstance(mape, (int, float)):
                model_text += f" | MAPE: {mape:.2f}%"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=model_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
    
    # Configurar diseño
    fig.update_layout(
        title='Pronóstico ARIMA con Intervalos de Confianza',
        xaxis_title='Fecha',
        yaxis_title='Número de Casos',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_interactive_epidemic_cycles(df_weekly, enhanced_ts_results):
    """
    Crea una visualización interactiva de ciclos epidémicos.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    if not isinstance(df_weekly.index, pd.DatetimeIndex):
        # Si el índice no es de tipo fecha, intentar convertirlo
        df_weekly = df_weekly.copy()
        if 'fecha' in df_weekly.columns:
            df_weekly.set_index('fecha', inplace=True)
        else:
            # No podemos proceder sin fechas
            fig = go.Figure()
            fig.add_annotation(
                text="No se pueden visualizar ciclos epidémicos sin información de fechas",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Análisis de Ciclos Epidémicos - No disponible")
            return fig
    
    epidemic_cycles = enhanced_ts_results.get('epidemic_cycles', {})
    
    # Si no hay ciclos detectados, mostrar mensaje
    if not epidemic_cycles.get('cycle_detected', False):
        fig = go.Figure()
        fig.add_annotation(
            text="No se detectaron ciclos epidémicos",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Análisis de Ciclos Epidémicos - No detectados")
        return fig
    
    # Crear gráfico base con datos originales
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_weekly.index,
        y=df_weekly['Casos'],
        mode='lines',
        name='Casos semanales',
        line=dict(color='blue', width=2),
        hovertemplate='%{x}<br>Casos: %{y}<extra></extra>'
    ))
    
    # Marcar brotes detectados
    outbreaks = enhanced_ts_results.get('outbreaks')
    outbreak_dates = []
    outbreak_values = []
    
    if isinstance(outbreaks, pd.DataFrame) and not outbreaks.empty:
        outbreak_dates = outbreaks.index
        outbreak_values = df_weekly.loc[outbreak_dates, 'Casos']
        
        fig.add_trace(go.Scatter(
            x=outbreak_dates,
            y=outbreak_values,
            mode='markers',
            name='Brotes detectados',
            marker=dict(size=12, color='red', symbol='star'),
            hovertemplate='Brote<br>%{x}<br>Casos: %{y}<extra></extra>'
        ))
        
        # Dibujar líneas verticales en cada brote
        for date in outbreak_dates:
            fig.add_shape(
                type="line",
                x0=date, y0=0,
                x1=date, y1=df_weekly['Casos'].max() * 0.9,
                line=dict(color="red", width=1, dash="dot"),
            )
    
    # Agregar información sobre el ciclo
    intervals = epidemic_cycles.get('intervals', {})
    cycle_length = epidemic_cycles.get('cycle_length')
    cycle_pattern = epidemic_cycles.get('cycle_pattern')
    confidence = epidemic_cycles.get('confidence')
    
    # Crear texto con información del ciclo
    pattern_desc = {
        'REGULAR': 'Regular',
        'MODERATELY_REGULAR': 'Moderadamente regular',
        'IRREGULAR': 'Irregular',
        'INSUFFICIENT_DATA': 'Datos insuficientes'
    }
    
    conf_desc = {
        'HIGH': 'Alta',
        'MEDIUM': 'Media',
        'LOW': 'Baja',
        'VERY_LOW': 'Muy baja'
    }
    
    cycle_text = f"<b>Patrón del Ciclo:</b> {pattern_desc.get(cycle_pattern, cycle_pattern)}<br>"
    cycle_text += f"<b>Intervalo medio:</b> {intervals.get('mean', 'N/A'):.1f} días<br>"
    cycle_text += f"<b>Desviación estándar:</b> {intervals.get('std', 'N/A'):.1f} días<br>"
    cycle_text += f"<b>Confianza:</b> {conf_desc.get(confidence, confidence)}"
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=cycle_text,
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Agregar próximo brote estimado si está disponible
    if 'next_outbreak_estimate' in epidemic_cycles:
        next_estimate = epidemic_cycles['next_outbreak_estimate']
        
        # Si hay datos sobre el intervalo de predicción
        if 'next_outbreak_interval' in epidemic_cycles:
            interval = epidemic_cycles['next_outbreak_interval']
            lower = interval['lower']
            upper = interval['upper']
            
            # Verificar si las fechas son de tipo datetime
            if hasattr(lower, 'to_pydatetime') and hasattr(upper, 'to_pydatetime'):
                # Añadir área sombreada para el intervalo de predicción
                fig.add_vrect(
                    x0=lower,
                    x1=upper,
                    fillcolor="rgba(255, 165, 0, 0.3)",
                    line_width=0,
                    annotation_text="Intervalo de predicción<br>del próximo brote",
                    annotation_position="top right",
                    annotation=dict(font_size=10)
                )
                
                # Añadir línea vertical para la estimación puntual
                fig.add_shape(
                    type="line",
                    x0=next_estimate, y0=0,
                    x1=next_estimate, y1=df_weekly['Casos'].max(),
                    line=dict(color="orange", width=2),
                )
                
                # Añadir estimación puntual
                fig.add_trace(go.Scatter(
                    x=[next_estimate],
                    y=[df_weekly['Casos'].max() * 0.75],
                    mode='markers+text',
                    marker=dict(size=14, color='orange', symbol='star'),
                    name='Próximo brote estimado',
                    text=["Próximo<br>brote"],
                    textposition="top center",
                    hovertemplate='Próximo brote estimado<br>%{x}<extra></extra>'
                ))
    
    # Configurar diseño
    fig.update_layout(
        title='Análisis de Ciclos Epidémicos y Predicción de Brotes',
        xaxis_title='Fecha',
        yaxis_title='Número de Casos',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_interactive_transmissibility(df_weekly, enhanced_ts_results):
    """
    Crea una visualización interactiva de las métricas de transmisibilidad (Rt).
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    if not isinstance(df_weekly.index, pd.DatetimeIndex):
        # Si el índice no es de tipo fecha, intentar convertirlo
        df_weekly = df_weekly.copy()
        if 'fecha' in df_weekly.columns:
            df_weekly.set_index('fecha', inplace=True)
        else:
            # No podemos proceder sin fechas
            fig = go.Figure()
            fig.add_annotation(
                text="No se pueden visualizar métricas de transmisibilidad sin información de fechas",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Métricas de Transmisibilidad (Rt) - No disponible")
            return fig
    
    transmissibility = enhanced_ts_results.get('transmissibility_metrics', {})
    
    if 'error' in transmissibility or 'rt_proxy' not in transmissibility:
        fig = go.Figure()
        fig.add_annotation(
            text="Métricas de transmisibilidad no disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Métricas de Transmisibilidad (Rt) - No disponible")
        return fig
    
    # Convertir proxy de Rt a Serie de pandas
    rt_proxy = transmissibility['rt_proxy']
    
    if isinstance(rt_proxy, dict):
        dates = [pd.to_datetime(date) if isinstance(date, str) else date 
                for date in rt_proxy.keys()]
        values = list(rt_proxy.values())
        
        rt_series = pd.Series(values, index=dates)
    else:
        rt_series = pd.Series(rt_proxy)
    
    # Crear gráfico base
    fig = go.Figure()
    
    # Añadir línea de Rt
    fig.add_trace(go.Scatter(
        x=rt_series.index,
        y=rt_series.values,
        mode='lines',
        name='Rt (transmisibilidad)',
        line=dict(color='purple', width=3),
        hovertemplate='%{x}<br>Rt: %{y:.2f}<extra></extra>'
    ))
    
    # Añadir línea horizontal de referencia en Rt=1
    fig.add_shape(
        type="line",
        x0=rt_series.index.min(),
        x1=rt_series.index.max(),
        y0=1, y1=1,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Añadir anotación para Rt=1
    fig.add_annotation(
        x=rt_series.index.min(),
        y=1,
        text="Rt = 1 (umbral de crecimiento)",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="red")
    )
    
    # Colorear áreas según nivel de transmisibilidad
    # Área de crecimiento (Rt > 1)
    growth_x = []
    growth_y = []
    
    for date, rt in zip(rt_series.index, rt_series.values):
        if rt > 1:
            growth_x.extend([date, date])
            growth_y.extend([1, rt])
    
    if growth_x:
        fig.add_trace(go.Scatter(
            x=growth_x,
            y=growth_y,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=True,
            name='Crecimiento (Rt > 1)'
        ))
    
    # Área de decrecimiento (Rt < 1)
    decrease_x = []
    decrease_y = []
    
    for date, rt in zip(rt_series.index, rt_series.values):
        if rt < 1:
            decrease_x.extend([date, date])
            decrease_y.extend([rt, 1])
    
    if decrease_x:
        fig.add_trace(go.Scatter(
            x=decrease_x,
            y=decrease_y,
            fill='toself',
            fillcolor='rgba(0, 128, 0, 0.2)',
            line=dict(color='rgba(0, 128, 0, 0)'),
            showlegend=True,
            name='Decrecimiento (Rt < 1)'
        ))
    
    # Agregar información sobre el Rt actual
    current_rt = transmissibility.get('current_rt')
    rt_mean = transmissibility.get('recent_rt_mean')
    above_threshold = transmissibility.get('above_threshold')
    
    if current_rt is not None:
        rt_status = "CRECIMIENTO EPIDÉMICO" if current_rt > 1 else "DECRECIMIENTO EPIDÉMICO"
        
        rt_text = f"<b>Rt actual:</b> {current_rt:.2f}<br>"
        rt_text += f"<b>Estado:</b> {rt_status}<br>"
        
        if rt_mean is not None:
            rt_text += f"<b>Rt promedio reciente:</b> {rt_mean:.2f}<br>"
        
        if above_threshold is not None:
            rt_text += f"<b>Tiempo con Rt>1:</b> {above_threshold:.0%}"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=rt_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
    
    # Configurar diseño
    fig.update_layout(
        title='Número Reproductivo Efectivo (Rt) - Métricas de Transmisibilidad',
        xaxis_title='Fecha',
        yaxis_title='Rt (transmisibilidad)',
        hovermode='x unified',
        yaxis=dict(
            range=[0, max(3, rt_series.max() * 1.1)]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_interactive_visualizations(df, analysis_type):
    """
    Crea todas las visualizaciones interactivas necesarias.
    """
    try:
        figs = {
            'Casos_Semanales': plot_interactive_weekly_cases(df),
            'Distribucion_por_edad': plot_interactive_age_distribution(df),
            'Mapa_Calor': plot_interactive_heatmap(df),
            'Diagnosticos': plot_diagnoses_treemap(df),
            'Comparacion_por_edades': plot_age_group_comparison(df),
            'Tendencia_Semanal': plot_weekly_trend(df),
            'Atenciones_Semanales_por_Edad': plot_interactive_weekly_cases_by_age_group(df)
        }
        
        # Añadir visualizaciones específicas según el tipo de análisis
        if analysis_type == 'respiratorio':
            figs.update({
                'Tipos_Infecciones_Respiratorias': plot_respiratory_types(df),
                'Evolucion_COVID': plot_covid_evolution(df)
            })
        elif analysis_type == 'gastrointestinal':
            figs.update({
                'Tipos_Infecciones_Gastrointestinales': plot_gastrointestinal_types(df),
                'Severidad_Casos': plot_severity_distribution(df)
            })
        elif analysis_type == 'varicela':
            figs.update({
                'Distribucion_Varicela': plot_varicela_distribution(df),
                'Severidad_Varicela': plot_varicela_severity(df)
            })
        elif analysis_type == 'manopieboca':
            figs.update({
                'Distribucion_ManoPieBoca': plot_manopieboca_distribution(df),
                'Severidad_ManoPieBoca': plot_manopieboca_severity(df)
            })
        
        # NUEVO: Añadir visualizaciones de análisis avanzado de series temporales
        try:
            # Preparar datos semanales para las visualizaciones
            df_weekly = prepare_data(df)
            df_weekly['fecha'] = df_weekly.apply(
                lambda row: datetime.strptime(f"{int(row['Año'])}-W{int(row['Semana Epidemiologica'])}-1", "%Y-W%W-%w"), 
                axis=1
            )
            df_weekly = df_weekly.sort_values('fecha').set_index('fecha')
            
            # Obtener resultados del análisis avanzado de series temporales
            # En un entorno real, estos resultados vendrían del análisis realizado en statistical_analysis.py
            # Aquí simulamos realizar un análisis básico para obtener los resultados
            try:
                from src.statistical_analysis import perform_advanced_analysis
                temp_results = perform_advanced_analysis(df, analysis_type)
                enhanced_ts_results = temp_results.get('enhanced_time_series', {})
            except:
                # Fallback: usar datos del dataframe para mostrar ejemplos más simples
                from src.statistical_analysis import enhanced_time_series_analysis
                enhanced_ts_results = enhanced_time_series_analysis(df_weekly)
            
            # Añadir visualizaciones solo si existen resultados
            if enhanced_ts_results:
                figs.update({
                    'Cambios_Estructurales': plot_interactive_structural_changes(df_weekly, enhanced_ts_results),
                    'Pronostico_ARIMA': plot_interactive_arima_forecast(df_weekly, enhanced_ts_results),
                    'Ciclos_Epidemicos': plot_interactive_epidemic_cycles(df_weekly, enhanced_ts_results),
                    'Transmisibilidad_Rt': plot_interactive_transmissibility(df_weekly, enhanced_ts_results)
                })
        except Exception as e:
            print(f"Advertencia: No se pudieron generar visualizaciones avanzadas de series temporales: {str(e)}")
        
        return figs
        
    except Exception as e:
        print(f"Error al crear visualizaciones interactivas: {str(e)}")
        return {}

# Funciones específicas para enfermedades respiratorias
def plot_respiratory_types(df):
    """
    Crea una visualización interactiva de tipos de infecciones respiratorias.
    """
    respiratory_types = df[df['CIE10 DP'].str.startswith('J')].groupby(['Año', 'CIE10 DP', 'Diagnostico Principal'])['CIE10 DP'].count().reset_index(name='Casos')
    
    fig = px.bar(respiratory_types,
                 x='CIE10 DP',
                 y='Casos',
                 color='Año',
                 title='Tipos de Infecciones Respiratorias por Año',
                 hover_data=['Diagnostico Principal'],
                 barmode='group')
    
    fig.update_layout(
        xaxis_title='Código CIE-10',
        yaxis_title='Número de Casos',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_covid_evolution(df):
    """
    Crea una visualización interactiva de la evolución de casos COVID-19.
    """
    covid_data = df[df['CIE10 DP'].str.startswith('U07')].copy()
    covid_weekly = []
    
    for year in covid_data['Año'].unique():
        df_year = covid_data[covid_data['Año'] == year]
        max_weeks = get_year_max_weeks(year)
        
        for week in range(1, max_weeks + 1):
            cases = df_year[df_year['Semana Epidemiologica'] == week]['CIE10 DP'].count()
            covid_weekly.append({
                'Año': year,
                'Semana': week,
                'Casos': cases
            })
    
    covid_df = pd.DataFrame(covid_weekly)
    
    fig = px.line(covid_df,
                  x='Semana',
                  y='Casos',
                  color='Año',
                  title='Evolución de Casos COVID-19 por Semana Epidemiológica',
                  markers=True)
    
    fig.update_layout(
        xaxis_title='Semana Epidemiológica',
        yaxis_title='Número de Casos COVID-19',
        hovermode='x unified'
    )
    
    return fig

# Funciones específicas para enfermedades gastrointestinales
def plot_gastrointestinal_types(df):
    """
    Crea una visualización interactiva de tipos de infecciones gastrointestinales.
    """
    gi_types = df[df['CIE10 DP'].str.startswith('A')].groupby(['Año', 'CIE10 DP', 'Diagnostico Principal'])['CIE10 DP'].count().reset_index(name='Casos')
    
    fig = px.bar(gi_types,
                 x='CIE10 DP',
                 y='Casos',
                 color='Año',
                 title='Tipos de Infecciones Gastrointestinales por Año',
                 hover_data=['Diagnostico Principal'],
                 barmode='group')
    
    fig.update_layout(
        xaxis_title='Código CIE-10',
        yaxis_title='Número de Casos',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_severity_distribution(df):
    """
    Crea una visualización interactiva de la distribución de severidad de casos.
    """
    severity_data = df.groupby(['Año', 'Destino'])['CIE10 DP'].count().reset_index(name='Casos')
    
    fig = px.sunburst(severity_data,
                      path=['Año', 'Destino'],
                      values='Casos',
                      title='Distribución de Severidad de Casos por Año')
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# Funciones específicas para varicela
def plot_varicela_distribution(df):
    """
    Crea una visualización interactiva de la distribución de casos de varicela.
    """
    varicela_weekly = []
    
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)
        
        for week in range(1, max_weeks + 1):
            cases = df_year[df_year['Semana Epidemiologica'] == week]['CIE10 DP'].count()
            age_mean = df_year[df_year['Semana Epidemiologica'] == week]['Edad_Anios'].mean()
            
            varicela_weekly.append({
                'Año': year,
                'Semana': week,
                'Casos': cases,
                'Edad_Promedio': age_mean
            })
    
    varicela_df = pd.DataFrame(varicela_weekly)
    
    fig = px.scatter(varicela_df,
                    x='Semana',
                    y='Casos',
                    size='Casos',
                    color='Año',
                    hover_data=['Edad_Promedio'],
                    title='Distribución de Casos de Varicela por Semana')
    
    fig.update_layout(
        xaxis_title='Semana Epidemiológica',
        yaxis_title='Número de Casos',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_varicela_severity(df):
    """
    Crea una visualización interactiva de la severidad de casos de varicela.
    """
    severity_by_age = df.groupby(['Año', 'Grupo_Edad', 'Destino'])['CIE10 DP'].count().reset_index(name='Casos')
    
    fig = px.sunburst(severity_by_age,
                      path=['Año', 'Grupo_Edad', 'Destino'],
                      values='Casos',
                      title='Severidad de Casos de Varicela por Grupo Etario y Año')
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# Funciones específicas para mano-pie-boca
def plot_manopieboca_distribution(df):
    """
    Crea una visualización interactiva de la distribución de casos de mano-pie-boca.
    """
    mpb_data = df.groupby(['Año', 'Mes', 'Grupo_Edad'])['CIE10 DP'].count().reset_index(name='Casos')
    
    fig = px.area(mpb_data,
                  x='Mes',
                  y='Casos',
                  color='Grupo_Edad',
                  facet_row='Año',
                  title='Distribución Mensual de Casos de Mano-Pie-Boca por Grupo Etario')
    
    fig.update_layout(
        xaxis_title='Mes',
        yaxis_title='Número de Casos',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_manopieboca_severity(df):
    """
    Crea una visualización interactiva de la severidad de casos de mano-pie-boca.
    """
    severity_data = df.groupby(['Año', 'Mes', 'Destino'])['CIE10 DP'].count().reset_index(name='Casos')
    
    fig = px.bar(severity_data,
                 x='Mes',
                 y='Casos',
                 color='Destino',
                 facet_row='Año',
                 title='Severidad de Casos de Mano-Pie-Boca por Mes y Año',
                 barmode='stack')
    
    fig.update_layout(
        xaxis_title='Mes',
        yaxis_title='Número de Casos',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

if __name__ == "__main__":
    print("Este módulo está diseñado para ser importado y utilizado en streamlit_dashboard.py")
    print("Para visualizar los gráficos interactivos, ejecute streamlit_dashboard.py")