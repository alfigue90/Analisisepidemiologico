import pandas as pd
import numpy as np
from scipy import stats
import os

def get_year_max_weeks(year):
    """
    Retorna el número máximo de semanas para un año específico.
    """
    return 53 if year == 2025 else 52

def generate_weekly_cases_by_age_group_section(results):
    weekly_cases_by_age_group = results.get('weekly_cases_by_age_group', {})
    report = "21. Análisis de atenciones semanales por rango etario:\n\n"

    for year, data in weekly_cases_by_age_group.items():
        report += f"  Año {year}:\n"
        total_cases = data.drop('Semana', axis=1).sum().sum()  # Suma de casos excluyendo la columna 'Semana'
        for group in ['Menor de 1 A', '1 a 4 A', '5 a 14 A', '15 a 64 A', '65 y más A']:
            group_total = data[group].sum()
            group_percentage = (group_total / total_cases) * 100 if total_cases > 0 else 0
            report += f"  - {group}: {group_total:.0f} casos ({group_percentage:.2f}%)\n"

        # Encontrar la semana con más atenciones (excluyendo la columna 'Semana')
        peak_week = data.drop('Semana', axis=1).sum(axis=1).idxmax() + 1 # Se suma uno porque las semanas se indexan desde 1
        report += f"  - Semana con más atenciones: {peak_week}\n\n"

    return report

def generate_enhanced_time_series_section(results):
    """
    Genera sección del informe para el análisis avanzado de series temporales.
    """
    enhanced_ts = results.get('enhanced_time_series', {})
    
    if not enhanced_ts:
        return "22. Análisis Avanzado de Series Temporales:\n  No disponible\n\n"
    
    report = "22. Análisis Avanzado de Series Temporales:\n\n"
    
    # a) Cambios estructurales
    structural_changes = enhanced_ts.get('structural_changes', {})
    change_points = structural_changes.get('change_points', [])
    
    report += "  a) Detección de Cambios Estructurales:\n"
    report += f"    Número de cambios detectados: {len(change_points)}\n"
    
    if 'change_dates' in structural_changes and structural_changes['change_dates']:
        report += "    Fechas de cambios estructurales:\n"
        for date in structural_changes['change_dates']:
            if hasattr(date, 'strftime'):
                report += f"      - {date.strftime('%Y-%m-%d')}\n"
            else:
                report += f"      - {date}\n"
    
    if 'segment_stats' in structural_changes:
        report += "    Estadísticas de segmentos:\n"
        for stat in structural_changes['segment_stats']:
            report += f"      - Cambio en {stat.get('change_date', stat.get('change_point', 'N/A'))}: "
            report += f"Cambio relativo: {stat.get('relative_change', 'N/A'):.2%}, "
            report += f"Significativo: {'Sí' if stat.get('significant', False) else 'No'}\n"
    
    # b) Modelo ARIMA
    report += "\n  b) Modelado ARIMA Avanzado:\n"
    
    if 'model_info' in enhanced_ts:
        model_info = enhanced_ts['model_info']
        report += f"    Orden ARIMA: {model_info.get('order', 'N/A')}\n"
        report += f"    Orden estacional: {model_info.get('seasonal_order', 'N/A')}\n"
        report += f"    AIC: {model_info.get('aic', 'N/A')}\n"
    
    if 'detected_seasonality' in enhanced_ts:
        seasonality = enhanced_ts['detected_seasonality']
        report += f"    Período estacional detectado: {seasonality.get('period', 'N/A')}\n"
    
    if 'model_fit' in enhanced_ts:
        model_fit = enhanced_ts['model_fit']
        report += f"    RMSE: {model_fit.get('rmse', 'N/A'):.2f}\n"
        if 'mape' in model_fit:
            report += f"    MAPE: {model_fit.get('mape', 'N/A'):.2f}%\n"
    
    # c) Pronóstico
    report += "\n  c) Pronóstico:\n"
    
    if 'forecast' in enhanced_ts and enhanced_ts['forecast'] is not None:
        forecast = enhanced_ts['forecast']
        forecast_df = forecast if isinstance(forecast, pd.DataFrame) else None
        
        if forecast_df is not None and not forecast_df.empty:
            # Mostrar solo primeras 5 semanas del pronóstico
            display_rows = min(5, len(forecast_df))
            
            for i in range(display_rows):
                date = forecast_df.index[i]
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                
                report += f"    {date_str}: {forecast_df['forecast'].iloc[i]:.1f} "
                report += f"(IC: {forecast_df['lower_ci'].iloc[i]:.1f} - {forecast_df['upper_ci'].iloc[i]:.1f})\n"
            
            if len(forecast_df) > display_rows:
                report += "    ...\n"
    else:
        report += "    Pronóstico no disponible\n"
    
    # d) Análisis de ciclos epidémicos
    report += "\n  d) Análisis de Ciclos Epidémicos:\n"
    
    cycles = enhanced_ts.get('epidemic_cycles', {})
    report += f"    Número de brotes detectados: {cycles.get('n_outbreaks', 'N/A')}\n"
    
    if cycles.get('cycle_detected', False):
        intervals = cycles.get('intervals', {})
        report += f"    Intervalo medio entre brotes: {intervals.get('mean', 'N/A'):.1f} días\n"
        report += f"    Desviación estándar: {intervals.get('std', 'N/A'):.1f} días\n"
        report += f"    Patrón de ciclo: {cycles.get('cycle_pattern', 'N/A')}\n"
        report += f"    Confianza: {cycles.get('confidence', 'N/A')}\n"
        
        if 'next_outbreak_estimate' in cycles:
            next_date = cycles['next_outbreak_estimate']
            date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
            report += f"    Próximo brote estimado: {date_str}\n"
            
            if 'next_outbreak_interval' in cycles:
                interval = cycles['next_outbreak_interval']
                lower = interval['lower'].strftime('%Y-%m-%d') if hasattr(interval['lower'], 'strftime') else str(interval['lower'])
                upper = interval['upper'].strftime('%Y-%m-%d') if hasattr(interval['upper'], 'strftime') else str(interval['upper'])
                report += f"    Intervalo de predicción: {lower} - {upper}\n"
        
        if 'seasonality' in cycles and cycles['seasonality'].get('detected', False):
            seasonality = cycles['seasonality']
            report += f"    Estacionalidad detectada: Mes dominante {seasonality.get('dominant_month', 'N/A')}\n"
            if 'seasonal_period' in seasonality:
                report += f"    Período estacional: {', '.join(map(str, seasonality['seasonal_period']))}\n"
    else:
        report += "    No se detectó un patrón cíclico claro\n"
    
    # e) Indicadores de alerta temprana
    report += "\n  e) Indicadores de Alerta Temprana:\n"
    
    early_warning = enhanced_ts.get('early_warning', {})
    
    if early_warning:
        current_trend = early_warning.get('current_trend')
        trend_change = early_warning.get('trend_change_percent')
        growth_rate = early_warning.get('growth_rate')
        
        if current_trend is not None:
            trend_direction = "creciente" if current_trend > 0 else "decreciente" if current_trend < 0 else "estable"
            report += f"    Tendencia actual: {trend_direction} ({current_trend:.2f} casos/semana)\n"
        
        if trend_change is not None:
            report += f"    Cambio en tendencia: {trend_change:.1f}%\n"
        
        if growth_rate is not None:
            report += f"    Tasa de crecimiento: {growth_rate:.1f}%\n"
        
        # Comparación con histórico
        historical_comp = early_warning.get('current_vs_historical', {})
        if 'status' in historical_comp:
            report += f"    Estado actual vs. histórico: {historical_comp.get('status', 'N/A')}\n"
            report += f"    Desviación del promedio histórico: {historical_comp.get('percent_deviation', 'N/A'):.1f}%\n"
    else:
        report += "    Indicadores no disponibles\n"
    
    # f) Métricas de transmisibilidad
    report += "\n  f) Métricas de Transmisibilidad:\n"
    
    transmissibility = enhanced_ts.get('transmissibility_metrics', {})
    
    if transmissibility and 'error' not in transmissibility:
        current_rt = transmissibility.get('current_rt')
        rt_mean = transmissibility.get('recent_rt_mean')
        
        if current_rt is not None:
            report += f"    Rt proxy actual: {current_rt:.2f}\n"
            rt_status = "CRECIMIENTO" if current_rt > 1 else "DECRECIMIENTO"
            report += f"    Estado de transmisión: {rt_status}\n"
        
        if rt_mean is not None:
            report += f"    Rt promedio reciente: {rt_mean:.2f}\n"
            above_threshold = transmissibility.get('above_threshold')
            if above_threshold is not None:
                report += f"    Proporción de tiempo con Rt>1: {above_threshold:.0%}\n"
    else:
        error_msg = transmissibility.get('error', 'Métricas no disponibles')
        report += f"    {error_msg}\n"
    
    report += "\n"
    return report

def generate_report(analysis_results, analysis_type):
    report = f"Informe de Análisis Epidemiológico - {analysis_type.capitalize()}\n\n"

    sections = [
        generate_trend_section,
        generate_outliers_section,
        generate_seasonality_section,
        generate_sarima_section,
        generate_outbreaks_section,
        generate_autocorrelation_section,
        generate_age_group_section,
        generate_establishment_section,
        generate_sex_section,
        generate_origin_country_section,
        generate_top_diagnoses_section,
        generate_specific_analysis_section,
        generate_proportions_section,
        generate_correlation_section,
        generate_time_series_section,
        generate_visualizations_description,
        generate_2024_detailed_section,
        generate_2025_detailed_section, # Se ha añadido la sección para 2025
        generate_establishment_analysis_section,
        generate_weekly_change_section,
        generate_interpretation_and_recommendations,
        generate_weekly_cases_by_age_group_section,
        generate_enhanced_time_series_section, # NUEVA SECCIÓN: Análisis avanzado de series temporales
    ]

    for section_func in sections:
        try:
            if section_func == generate_specific_analysis_section:
                report += section_func(analysis_results, analysis_type)
            elif section_func in [generate_visualizations_description, generate_interpretation_and_recommendations]:
                report += section_func(analysis_type, analysis_results)
            else:
                report += section_func(analysis_results)
        except Exception as e:
            report += f"Error al generar la sección {section_func.__name__}: {str(e)}\n\n"

    return report

def save_report(report, analysis_type):
    """
    Guarda el informe en un archivo.

    Args:
        report (str): Contenido del informe
        analysis_type (str): Tipo de análisis realizado

    Returns:
        str: Ruta del archivo guardado
    """
    os.makedirs('analisis', exist_ok=True)
    output_path = os.path.join('analisis', f'informe_epidemiologico_{analysis_type}.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return output_path

def generate_trend_section(results):
    trend = results.get('trend', 'No disponible')
    trend_pvalue = results.get('trend_pvalue', 'No disponible')
    ci = results.get('trend_confidence_interval', ['No disponible', 'No disponible'])

    report = f"1. Tendencia general:\n"
    report += f"  Coeficiente de tendencia: {trend}\n"
    report += f"  Valor p de la tendencia: {trend_pvalue}\n"
    report += f"  Intervalo de confianza (95%): [{ci[0]}, {ci[1]}]\n"
    if isinstance(trend_pvalue, float) and trend_pvalue < 0.05:
        report += "  La tendencia es estadísticamente significativa.\n"
    elif isinstance(trend_pvalue, float):
        report += "  La tendencia no es estadísticamente significativa.\n"
    else:
        report += "  No se pudo determinar la significancia estadística de la tendencia.\n"
    report += "\n"
    return report

def generate_outliers_section(results):
    outliers = results.get('outliers', pd.DataFrame())
    report = f"2. Detección de outliers:\n"
    report += f"  Número de outliers detectados: {len(outliers)}\n"
    if len(outliers) > 0:
        report += "  Fechas de los outliers:\n"
        for date, row in outliers.iterrows():
            report += f"  - {date.strftime('%Y-%m-%d')}: {row['Casos']} casos\n"
    report += "\n"
    return report

def generate_seasonality_section(results):
    seasonal_component = results.get('seasonal_component')
    report = f"3. Análisis de estacionalidad:\n"
    if seasonal_component is not None:
        # Verificar si 'seasonal' es una Serie de pandas y tiene el método 'max'
        if hasattr(seasonal_component, 'seasonal') and hasattr(seasonal_component.seasonal, 'max'):
            report += f"  Componente estacional máximo: {seasonal_component.seasonal.max():.4f}\n"
            report += f"  Componente estacional mínimo: {seasonal_component.seasonal.min():.4f}\n"
        else:
            report += "  Los datos de estacionalidad no están en el formato esperado.\n"
    else:
        report += "  Análisis de estacionalidad no disponible\n"
    report += "\n"
    return report

def generate_sarima_section(results):
    sarima_model = results.get('sarima_model')
    report = f"4. Modelado SARIMA:\n"
    if sarima_model is not None:
        report += f"  AIC del modelo SARIMA: {sarima_model.aic:.4f}\n"
    else:
        report += "  Modelado SARIMA no disponible\n"
    report += "\n"
    return report

def generate_outbreaks_section(results):
    outbreaks = results.get('outbreaks', pd.DataFrame())
    report = f"5. Detección de brotes:\n"
    report += f"  Número de brotes detectados: {len(outbreaks)}\n"
    if len(outbreaks) > 0:
        report += "  Fechas de los brotes:\n"
        for date, row in outbreaks.iterrows():
            report += f"  - {date.strftime('%Y-%m-%d')}: {row['Casos']} casos\n"
    report += "\n"
    return report

def generate_autocorrelation_section(results):
    acf = results.get('acf', [])
    report = f"6. Análisis de autocorrelación:\n"
    significant_lag = np.where(np.abs(acf) > 0.05)[0][1:6] if len(np.where(np.abs(acf) > 0.05)[0]) > 1 else "No significativo"
    report += f"  Autocorrelación significativa hasta el lag: {significant_lag}\n"
    report += "\n"
    return report

def generate_age_group_section(results):
    age_group_analysis = results.get('grupo_edad_analysis', pd.DataFrame())
    report = "7. Análisis por grupo de edad:\n"
    if 'Casos' in age_group_analysis.columns and 'Grupo_Edad' in age_group_analysis.columns:
        for _, row in age_group_analysis.iterrows():
            report += f"  {row['Grupo_Edad']}: {row['Casos']} casos (Edad promedio: {row['Edad_Promedio']:.2f})\n"
    else:
        report += "  No hay datos disponibles para el análisis por grupo de edad.\n"
    report += "\n"
    return report

def generate_establishment_section(results):
    establishment_analysis = results.get('estableciemiento_analysis', pd.DataFrame())
    report = "8. Análisis por establecimiento (Top 5):\n"
    if 'Casos' in establishment_analysis.columns and 'Estableciemiento' in establishment_analysis.columns:
        for _, row in establishment_analysis.nlargest(5, 'Casos').iterrows():
            report += f"  {row['Estableciemiento']}: {row['Casos']} casos (Edad promedio: {row['Edad_Promedio']:.2f})\n"
    else:
        report += "  No hay datos disponibles para el análisis por establecimiento.\n"
    report += "\n"
    return report

def generate_sex_section(results):
    sex_analysis = results.get('sexo_analysis', pd.DataFrame())
    report = "9. Análisis por sexo:\n"
    if 'Casos' in sex_analysis.columns and 'Sexo' in sex_analysis.columns:
        for _, row in sex_analysis.iterrows():
            report += f"  {row['Sexo']}: {row['Casos']} casos (Edad promedio: {row['Edad_Promedio']:.2f})\n"
    else:
        report += "  No hay datos disponibles para el análisis por sexo.\n"
    report += "\n"
    return report

def generate_origin_country_section(results):
    origin_analysis = results.get('pais_origen_analysis', pd.DataFrame())
    report = "10. Análisis por país de origen (Top 5):\n"
    if 'Casos' in origin_analysis.columns and 'Pais_Origen' in origin_analysis.columns:
        for _, row in origin_analysis.nlargest(5, 'Casos').iterrows():
            report += f"  {row['Pais_Origen']}: {row['Casos']} casos (Edad promedio: {row['Edad_Promedio']:.2f})\n"
    else:
        report += "  No hay datos disponibles para el análisis por país de origen.\n"
    report += "\n"
    return report

def generate_top_diagnoses_section(results):
    top_10_diagnoses = results.get('top_10_diagnoses', pd.Series())
    report = "11. Top 10 diagnósticos:\n"
    for diagnosis, count in top_10_diagnoses.items():
        report += f"  {diagnosis}: {count} casos\n"
    report += "\n"
    return report

def generate_specific_analysis_section(results, analysis_type):
    if analysis_type == 'respiratorio':
        return generate_respiratory_report(results.get('respiratorio_specific', {}))
    elif analysis_type == 'gastrointestinal':
        return generate_gastrointestinal_report(results.get('gastrointestinal_specific', {}))
    elif analysis_type == 'varicela':
        return generate_varicela_report(results.get('varicela_specific', {}))
    elif analysis_type == 'manopieboca':
        return generate_manopieboca_report(results.get('manopieboca_specific', {}))
    return "No se encontró un análisis específico para el tipo de enfermedad seleccionado.\n\n"

def generate_respiratory_report(respiratory_results):
    report = "12. Análisis específico de enfermedades respiratorias:\n"
    if not respiratory_results:
        return report + "  No hay datos específicos disponibles para enfermedades respiratorias.\n\n"

    report += f"  a) Tipos de infecciones respiratorias más comunes:\n"
    for infection_type, count in respiratory_results.get('infection_types', {}).items():
        report += f"    - {infection_type}: {count} casos\n"
    report += f"  b) Casos de influenza: {respiratory_results.get('influenza_cases', 'No disponible')}\n"
    report += f"  c) Casos de otras infecciones respiratorias: {respiratory_results.get('other_respiratory_cases', 'No disponible')}\n"
    report += f"  d) Casos de neumonía: {respiratory_results.get('pneumonia_cases', 'No disponible')}\n"
    report += f"  e) Casos de COVID-19: {respiratory_results.get('covid19_cases', 'No disponible')}\n"
    report += "\n"
    return report

def generate_gastrointestinal_report(gastrointestinal_results):
    if not gastrointestinal_results:
        return "12. Análisis específico de enfermedades gastrointestinales:\n  No hay datos disponibles.\n\n"

    report = "12. Análisis específico de enfermedades gastrointestinales:\n"
    
    # Secciones existentes
    report += f"  a) Total de casos gastrointestinales: {gastrointestinal_results.get('total_cases', 'No disponible')}\n"
    report += f"  b) Casos de diarrea infecciosa: {gastrointestinal_results.get('infectious_diarrhea_cases', 'No disponible')}\n"
    report += f"  c) Casos de diarrea no infecciosa: {gastrointestinal_results.get('non_infectious_diarrhea_cases', 'No disponible')}\n"
    report += f"  d) Casos de intoxicación alimentaria: {gastrointestinal_results.get('food_poisoning_cases', 'No disponible')}\n"
    report += f"  e) Casos de gastroenteritis viral: {gastrointestinal_results.get('viral_gastroenteritis_cases', 'No disponible')}\n\n"

    # Nueva sección de análisis por comuna
    comuna_analysis = gastrointestinal_results.get('comuna_analysis', {})
    if comuna_analysis:
        report += "  f) Análisis por Comuna:\n\n"
        
        # Análisis anual
        annual_dist = comuna_analysis.get('annual_distribution', {})
        if annual_dist:
            report += "    Distribución anual:\n"
            for year in sorted(annual_dist.keys()):
                report += f"      Año {year}:\n"
                for comuna, casos in annual_dist[year].items():
                    report += f"        - {comuna}: {int(casos)} casos\n"
            report += "\n"
        
        # Análisis mensual
        monthly_dist = comuna_analysis.get('monthly_distribution', {})
        if monthly_dist:
            report += "    Distribución mensual:\n"
            for year in sorted(monthly_dist.keys()):
                report += f"      Año {year}:\n"
                for month in range(1, 13):
                    if month in monthly_dist[year]:
                        report += f"        Mes {month}:\n"
                        for comuna, casos in monthly_dist[year][month].items():
                            if casos > 0:
                                report += f"          - {comuna}: {int(casos)} casos\n"
            report += "\n"
        
        # Añadir un resumen de las comunas más afectadas
        if annual_dist:
            report += "    Resumen de comunas más afectadas:\n"
            total_by_comuna = {}
            for year_data in annual_dist.values():
                for comuna, casos in year_data.items():
                    total_by_comuna[comuna] = total_by_comuna.get(comuna, 0) + casos
            
            for comuna, total in sorted(total_by_comuna.items(), key=lambda x: x[1], reverse=True):
                report += f"      - {comuna}: {int(total)} casos totales\n"

    report += "\n"
    return report

def generate_varicela_report(varicela_results):
    report = "12. Análisis específico de casos de varicela:\n"
    if not varicela_results:
        return report + "  No hay datos específicos disponibles para casos de varicela.\n\n"

    report += f"  a) Total de casos de varicela: {varicela_results.get('total_cases', 'No disponible')}\n"
    report += "  b) Distribución por edad:\n"
    for stat, value in varicela_results.get('age_distribution', {}).items():
        report += f"    - {stat}: {value:.2f}\n"
    report += "  c) Distribución por género:\n"
    for gender, proportion in varicela_results.get('gender_distribution', {}).items():
        report += f"    - {gender}: {proportion:.2%}\n"
    report += "  d) Distribución mensual:\n"
    for month, cases in varicela_results.get('monthly_pattern', {}).items():
        report += f"    - Mes {month}: {cases} casos\n"
    report += f"  e) Tendencia anual:\n"
    for year, cases in varicela_results.get('yearly_trend', {}).items():
        report += f"    - Año {year}: {cases} casos\n"
    report += f"  f) Severidad (porcentaje de casos no domiciliarios): {varicela_results.get('severity', 'No disponible'):.2%}\n"
    report += "\n"
    return report

def generate_manopieboca_report(manopieboca_results):
    report = "12. Análisis específico de casos de mano-pie-boca:\n"
    if not manopieboca_results:
        return report + "  No hay datos específicos disponibles para casos de mano-pie-boca.\n\n"

    report += f"  a) Total de casos de mano-pie-boca: {manopieboca_results.get('total_cases', 'No disponible')}\n"
    report += "  b) Distribución por edad:\n"
    for stat, value in manopieboca_results.get('age_distribution', {}).items():
        report += f"    - {stat}: {value:.2f}\n"
    report += "  c) Distribución por género:\n"
    for gender, proportion in manopieboca_results.get('gender_distribution', {}).items():
        report += f"    - {gender}: {proportion:.2%}\n"
    report += "  d) Distribución mensual:\n"
    for month, cases in manopieboca_results.get('monthly_pattern', {}).items():
        report += f"    - Mes {month}: {cases} casos\n"
    report += f"  e) Tendencia anual:\n"
    for year, cases in manopieboca_results.get('yearly_trend', {}).items():
        report += f"    - Año {year}: {cases} casos\n"
    report += f"  f) Severidad (porcentaje de casos no domiciliarios): {manopieboca_results.get('severity', 'No disponible'):.2%}\n"
    report += "\n"
    return report

def generate_proportions_section(results):
    proportion_analysis = results.get('proportion_analysis', {})
    report = "13. Análisis de proporciones:\n"
    for category, stats in proportion_analysis.items():
        # Asegurarse de que 'percentage' esté disponible y redondear el valor
        percentage = round(stats.get('percentage', 0), 2)
        report += f"  {category}: {percentage}%"
        
        # Añadir intervalo de confianza si está disponible
        if 'ci_lower' in stats and 'ci_upper' in stats:
            ci_lower = round(stats['ci_lower'], 2)
            ci_upper = round(stats['ci_upper'], 2)
            report += f" (IC 95%: {ci_lower}% - {ci_upper}%)"
        
        report += "\n"
    report += "\n"
    return report

def generate_correlation_section(results):
    correlation_analysis = results.get('correlation_analysis', {})
    report = "14. Análisis de correlación:\n"
    for var1, corr_data in correlation_analysis.items():
        for var2, corr_value in corr_data.items():
            if var1 != var2:
                report += f"  Correlación entre {var1} y {var2}: {corr_value:.4f}\n"
    report += "\n"
    return report

def generate_time_series_section(results):
    time_series_analysis = results.get('time_series_analysis', {})
    report = "15. Análisis de series temporales:\n"
    adf_test = time_series_analysis.get('adf_test', {})
    report += f"  Test de estacionariedad (ADF):\n"
    report += f"  - Estadístico ADF: {adf_test.get('adf_statistic', 'No disponible'):.4f}\n"
    report += f"  - Valor p: {adf_test.get('p_value', 'No disponible'):.4f}\n"
    report += "  La serie temporal es " + ("estacionaria" if adf_test.get('p_value', 1) < 0.05 else "no estacionaria") + "\n"
    report += "\n"
    return report

def generate_visualizations_description(analysis_type, analysis_results=None):
    report = "16. Visualizaciones generadas:\n"
    report += "  a) Gráfico de barras de los 10 diagnósticos más frecuentes por año.\n"
    report += "  b) Gráfico de barras apiladas de casos semanales por grupo etario para cada año.\n"
    report += "  c) Heatmap de casos por mes y año.\n"
    report += "  d) Gráfico de densidad de la distribución de casos por sexo y edad.\n"
    report += "  e) Gráfico de barras de pacientes derivados a centros de mayor complejidad por año.\n"
    report += "  f) Gráfico de líneas de casos semanales por año.\n"
    report += "  g) Gráfico de tendencia con intervalos de confianza.\n"
    report += "  h) Gráfico de detección de brotes.\n"
    report += "  i) Gráficos de autocorrelación y autocorrelación parcial.\n"
    report += "  j) Gráfico de barras de análisis de proporciones.\n"
    report += "  k) Gráfico de líneas comparativo de casos por establecimiento.\n"
    report += "  l) Gráfico de barras de análisis de cambio porcentual semanal.\n"
    report += "  m) Gráfico de cambios estructurales en la serie temporal.\n"  # NUEVO
    report += "  n) Gráfico de pronóstico ARIMA con intervalos de confianza.\n"  # NUEVO
    report += "  o) Gráfico de ciclos epidémicos y estimación del próximo brote.\n"  # NUEVO
    report += "  p) Gráfico de número reproductivo efectivo (Rt).\n"  # NUEVO

    # Verificar si analysis_results tiene la clave 'detailed_2025_analysis'
    if analysis_results and 'detailed_2025_analysis' in analysis_results:
        # Verificar si 'disease_specific' está en 'detailed_2025_analysis'
        if 'disease_specific' in analysis_results['detailed_2025_analysis']:
            # Acceder a la información específica de la enfermedad para 2025
            disease_specific_2025 = analysis_results['detailed_2025_analysis']['disease_specific']

            if analysis_type == 'varicela' and 'varicela' in disease_specific_2025:
                report += "  q) Gráfico de distribución de edad para casos de varicela.\n"
                report += "  r) Gráfico circular de severidad de casos de varicela.\n"
                report += "  s) Gráfico de barras de distribución mensual de casos de varicela.\n"
                report += "  t) Gráfico de líneas de tendencia anual de casos de varicela.\n"
            elif analysis_type == 'manopieboca' and 'manopieboca' in disease_specific_2025:
                report += "  q) Gráfico de distribución de edad para casos de mano-pie-boca.\n"
                report += "  r) Gráfico circular de severidad de casos de mano-pie-boca.\n"
                report += "  s) Gráfico de barras de distribución mensual de casos de mano-pie-boca.\n"
                report += "  t) Gráfico de líneas de tendencia anual de casos de mano-pie-boca.\n"

    report += "\n"
    return report

def generate_2024_detailed_section(results):
    detailed_2024 = results.get('detailed_2024_analysis', {})
    if not detailed_2024:
        return "17. Análisis Detallado del Año 2024:\n  No hay datos disponibles para el análisis detallado de 2024.\n\n"

    report = "17. Análisis Detallado del Año 2024:\n"
    report += f"  Total de casos en 2024: {detailed_2024.get('total_cases', 'No disponible')}\n\n"

    report += "  a) Distribución semanal de casos:\n"
    for week, cases in detailed_2024.get('weekly_distribution', {}).items():
        report += f"    Semana {week}: {cases} casos\n"

    report += "\n  b) Distribución por edad:\n"
    for stat, value in detailed_2024.get('age_distribution', {}).items():
        report += f"    {stat}: {value:.2f}\n"

    report += "\n  c) Distribución por género:\n"
    for gender, proportion in detailed_2024.get('gender_distribution', {}).items():
        report += f"    {gender}: {proportion:.2%}\n"

    report += "\n  d) Top 10 diagnósticos:\n"
    for diagnosis, count in detailed_2024.get('top_10_diagnoses', {}).items():
        report += f"    {diagnosis}: {count} casos\n"

    report += "\n  e) Tendencia mensual:\n"
    for month, cases in detailed_2024.get('monthly_trend', {}).items():
        report += f"    Mes {month}: {cases} casos\n"

    report += "\n  f) Análisis de severidad:\n"
    for destination, proportion in detailed_2024.get('severity_analysis', {}).items():
        report += f"    {destination}: {proportion:.2%}\n"

    report += "\n"
    return report

def generate_2025_detailed_section(results):
    detailed_2025 = results.get('detailed_2025_analysis', {})
    if not detailed_2025:
        return "18. Análisis Detallado del Año 2025:\n  No hay datos disponibles para el análisis detallado de 2025.\n\n"

    report = "18. Análisis Detallado del Año 2025 (53 semanas):\n\n"

    # Análisis general
    report += f"    - Media semanal: {detailed_2025.get('weekly_mean', 'No disponible'):.2f} casos\n"
    report += f"    - Mediana semanal: {detailed_2025.get('weekly_median', 'No disponible'):.2f} casos\n"
    report += f"    - Desviación estándar semanal: {detailed_2025.get('weekly_std', 'No disponible'):.2f}\n\n"

    # Distribución semanal
    report += "  b) Distribución semanal de casos:\n"
    weekly_distribution = detailed_2025.get('weekly_distribution', {})
    for week, data in weekly_distribution.items():
        cases = data.get('cases', 'No disponible')
        report += f"    Semana {week}: {cases} casos\n"
    report += "\n"

    # Distribución por edad
    report += "  c) Distribución por edad:\n"
    age_stats = detailed_2025.get('age_distribution', {})
    for stat, value in age_stats.items():
        report += f"    {stat}: {value:.2f}\n"
    report += "\n"

    # Distribución por género
    report += "  d) Distribución por género:\n"
    gender_dist = detailed_2025.get('gender_distribution', {})
    for gender, prop in gender_dist.items():
        report += f"    {gender}: {prop:.2%}\n"
    report += "\n"

    # Top diagnósticos
    report += "  e) Top 10 diagnósticos:\n"
    top_diagnoses = detailed_2025.get('top_10_diagnoses', {})
    for diagnosis, count in top_diagnoses.items():
        report += f"    {diagnosis}: {count} casos\n"
    report += "\n"

    # Tendencia mensual
    report += "  f) Tendencia mensual:\n"
    monthly_trend = detailed_2025.get('monthly_trend', {})
    for month, cases in monthly_trend.items():
        report += f"    Mes {month}: {cases} casos\n"
    report += "\n"

    # Análisis de severidad
    report += "  g) Análisis de severidad:\n"
    severity = detailed_2025.get('severity_analysis', {})
    for destination, prop in severity.items():
        report += f"    {destination}: {prop:.2%}\n"
    report += "\n"

    # Análisis específico según tipo de enfermedad
    if 'disease_specific' in detailed_2025:
        report += "  h) Análisis específico de la enfermedad:\n"
        disease_specific = detailed_2025['disease_specific']

        if 'respiratory' in disease_specific:
            resp = disease_specific['respiratory']
            report += f"    - Casos de influenza: {resp.get('influenza_cases', 0)}\n"
            report += f"    - Casos de neumonía: {resp.get('pneumonia_cases', 0)}\n"
            report += f"    - Casos de COVID-19: {resp.get('covid19_cases', 0)}\n"
        elif 'gastrointestinal' in disease_specific:
            gi = disease_specific['gastrointestinal']
            report += f"    - Casos de diarrea infecciosa: {gi.get('infectious_diarrhea_cases', 0)}\n"
            report += f"    - Casos de diarrea no infecciosa: {gi.get('non_infectious_diarrhea_cases', 0)}\n"
        elif 'varicela' in disease_specific:
            var = disease_specific['varicela']
            report += f"    - Tasa de complicaciones: {var.get('complication_rate', 0):.2%}\n"
            report += f"    - Casos en menores de 5 años: {var.get('under_5_cases', 0)}\n"
        elif 'manopieboca' in disease_specific:
            mpb = disease_specific['manopieboca']
            report += f"    - Tasa de complicaciones: {mpb.get('complication_rate', 0):.2%}\n"
            report += f"    - Casos en menores de 5 años: {mpb.get('under_5_cases', 0)}\n"

    # Comparación con años anteriores
    report += "  i) Comparación con años anteriores:\n"
    comparison = detailed_2025.get('year_comparison', {})
    for data in comparison:
        report += f"    Año: {data.get('year')}, Total de Casos: {data.get('total_cases')}, Promedio Semanal: {data.get('weekly_average')}, Diferencia Porcentual: {data.get('percent_difference')}\n"
    
    # Semana 53 específicamente
    report += "\n  j) Análisis específico de la Semana 53:\n"
    week_53 = detailed_2025.get('week_53_analysis', {})
    report += f"    - Total de casos: {week_53.get('total_cases', 0)}\n"
    report += f"    - Distribución por edad:\n"
    for stat, value in week_53.get('age_distribution', {}).items():
        report += f"      {stat}: {value:.2f}\n"
    report += f"    - Distribución por género:\n"
    for gender, proportion in week_53.get('gender_distribution', {}).items():
        report += f"      {gender}: {proportion:.2%}\n"
    report += f"    - Casos severos: {week_53.get('severe_cases', 0)}\n"
    report += "\n"
    return report

def generate_establishment_analysis_section(results):
    establishment_analysis = results.get('establishment_analysis', {})
    if not establishment_analysis:
        return "19. Análisis por Establecimiento:\n  No hay datos disponibles para el análisis por establecimiento.\n\n"

    report = "19. Análisis por Establecimiento:\n"

    for year, establishments in establishment_analysis.items():
        report += f"  Año {year}:\n"
        for estab, data in establishments.items():
            report += f"    {estab}:\n"
            report += f"      Total de casos: {data.get('total_cases', 'No disponible')}\n"
            report += f"      Edad promedio: {data.get('avg_age', 'No disponible'):.2f}\n"
            report += f"      Tasa de severidad: {data.get('severity_rate', 'No disponible'):.2f}%\n"
            report += "      Casos por semana:\n"
            weekly_cases = data.get('weekly_cases', {})
            for week, cases in weekly_cases.items():
                report += f"        Semana {week}: {cases} casos\n"
            report += "\n"

    return report

def generate_weekly_change_section(results):
    weekly_change = results.get('weekly_change', {})
    report = "20. Análisis de cambio semanal:\n"
    if not weekly_change:
        report += "  No hay datos disponibles para el análisis de cambio semanal.\n\n"
        return report

    for year, changes in weekly_change.items():
        report += f"  Año {year}:\n"
        # Verificar si 'changes' es un diccionario y no está vacío
        if isinstance(changes, dict) and changes:
            max_weeks = get_year_max_weeks(year)
            for week in range(1, max_weeks + 1):
                # Obtener el cambio para la semana actual, o 'No disponible' si no existe
                change = changes.get(week, 'No disponible')
                # Verificar si el cambio es un número antes de formatearlo
                if isinstance(change, (int, float)):
                    report += f"    Semana {week}: {change:.2f}%\n"
                else:
                    report += f"    Semana {week}: {change}\n"
        else:
            report += "    No hay datos de cambio semanal disponibles para este año.\n"
        report += "\n"

    return report

def generate_interpretation_and_recommendations(analysis_type, analysis_results):
    report = "21. Interpretación y recomendaciones:\n"
    report += "Basándose en el análisis anterior, se pueden extraer las siguientes conclusiones y recomendaciones:\n\n"
    report += "1. Tendencias temporales: [Interpretar la tendencia general y la estacionalidad]\n"
    report += "2. Grupos de riesgo: [Identificar los grupos de edad o demográficos más afectados]\n"
    report += "3. Diagnósticos principales: [Comentar sobre los diagnósticos más frecuentes y sus implicaciones]\n"
    report += "4. Recomendaciones para la gestión de recursos: [Sugerir ajustes basados en los patrones observados]\n"
    report += "5. Propuestas de intervención: [Sugerir medidas preventivas o de control basadas en los hallazgos]\n"
    report += f"6. Consideraciones específicas para {analysis_type}: [Añadir recomendaciones específicas]\n"
    report += "7. Futuras investigaciones: [Identificar áreas que requieren mayor estudio o seguimiento]\n"
    report += "8. Comparación con años anteriores (2021-2025): [Destacar diferencias o similitudes notables]\n"
    report += "9. Impacto de las intervenciones: [Evaluar el efecto de medidas implementadas, si las hubo]\n"
    report += "10. Preparación para futuros brotes: [Sugerir medidas de preparación basadas en los patrones observados]\n"

    # Añadir interpretaciones específicas para 2024
    report += "11. Análisis específico del año 2024:\n"
    detailed_2024 = analysis_results.get('detailed_2024_analysis', {})
    if detailed_2024:
        report += f"    - Total de casos en 2024: {detailed_2024.get('total_cases', 'No disponible')}\n"
        report += f"    - Tendencia observada: [Describir la tendencia mensual o semanal]\n"
        report += f"    - Grupos de edad más afectados: [Basado en la distribución por edad]\n"
        report += f"    - Diagnósticos predominantes: [Basado en el top 10 de diagnósticos]\n"
        if analysis_type == 'respiratorio':
            report += f"    - Proporción de casos de COVID-19: {detailed_2024.get('covid19_cases', 0) / detailed_2024.get('total_cases', 1):.2%}\n"
        elif analysis_type == 'gastrointestinal':
            report += f"    - Proporción de casos de diarrea infecciosa: {detailed_2024.get('infectious_diarrhea_cases', 0) / detailed_2024.get('total_cases', 1):.2%}\n"
        elif analysis_type == 'varicela':
            report += f"    - Severidad de los casos de varicela: {detailed_2024.get('severity', 'No disponible'):.2%}\n"
        elif analysis_type == 'manopieboca':
            report += f"    - Severidad de los casos de mano-pie-boca: {detailed_2024.get('severity', 'No disponible'):.2%}\n"
    else:
        report += "    No hay datos suficientes para realizar un análisis detallado del año 2024.\n"

    # NUEVO: Añadir interpretaciones basadas en el análisis avanzado de series temporales
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    if enhanced_ts:
        report += "\n12. Hallazgos del análisis avanzado de series temporales:\n"
        
        # Ciclos epidémicos
        cycles = enhanced_ts.get('epidemic_cycles', {})
        if cycles.get('cycle_detected', False):
            report += f"    - Se ha detectado un patrón cíclico {cycles.get('cycle_pattern', '').lower()} con un intervalo medio de {cycles.get('intervals', {}).get('mean', 0):.1f} días entre brotes.\n"
            
            if 'next_outbreak_estimate' in cycles:
                next_date = cycles['next_outbreak_estimate']
                date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
                report += f"    - Próximo brote estimado para: {date_str} (confianza: {cycles.get('confidence', 'BAJA')})\n"
                
            if 'seasonality' in cycles and cycles['seasonality'].get('detected', False):
                month = cycles['seasonality'].get('dominant_month', '')
                month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                              'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                if 1 <= month <= 12:
                    month_name = month_names[month-1]
                    report += f"    - Se ha detectado estacionalidad con mayor incidencia en {month_name}.\n"
        
        # Transmisibilidad
        transmissibility = enhanced_ts.get('transmissibility_metrics', {})
        if 'current_rt' in transmissibility:
            rt = transmissibility['current_rt']
            if rt > 1:
                report += f"    - El número reproductivo estimado (Rt={rt:.2f}) sugiere un crecimiento activo de casos.\n"
            else:
                report += f"    - El número reproductivo estimado (Rt={rt:.2f}) sugiere una fase de decrecimiento.\n"
        
        # Alerta temprana
        early_warning = enhanced_ts.get('early_warning', {})
        historical_comp = early_warning.get('current_vs_historical', {})
        if 'status' in historical_comp:
            status = historical_comp['status']
            deviation = historical_comp.get('percent_deviation', 0)
            
            if status == "ALERT":
                report += f"    - ALERTA: La situación actual está {deviation:.1f}% por encima del promedio histórico.\n"
            elif status == "WARNING":
                report += f"    - ADVERTENCIA: La situación actual está {deviation:.1f}% por encima del promedio histórico.\n"
            elif status == "BELOW_AVERAGE":
                report += f"    - Situación favorable: Los casos actuales están {abs(deviation):.1f}% por debajo del promedio histórico.\n"

    report += "\nNota: Este informe ha sido generado automáticamente basándose en los datos analizados. "
    report += "Se recomienda la revisión y validación por parte de expertos en epidemiología y salud pública "
    report += "antes de tomar decisiones basadas en estas conclusiones.\n"
    return report

if __name__ == "__main__":
    print("Este módulo está diseñado para ser importado y utilizado en main.py")
    print("Para generar un informe, ejecute main.py")