import os
import pandas as pd
import logging
from src.data_processing import load_and_preprocess_data
from src.statistical_analysis import perform_advanced_analysis
from src.report_generation import generate_report, save_report
from src.visualization import create_visualizations
from src.gpt_analysis import generate_gpt_analysis, save_gpt_analysis
from src.interactive_visualizations import create_interactive_visualizations

def get_year_max_weeks(year):
    """
    Obtiene el número máximo de semanas para un año específico.
    """
    return 53 if year == 2025 else 52

def setup_logging():
    """
    Configura el sistema de logging.
    """
    os.makedirs('analisis', exist_ok=True)
    logging.basicConfig(
        filename='analisis/log_analisis_epidemiologico.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Configurar logging
    setup_logging()

    # Configurar la ruta de los datos
    data_path = os.path.join('data')

    # Mostrar mensaje de bienvenida con nuevas capacidades
    print("\n============================================================")
    print("Sistema de Análisis Epidemiológico SAR-SAPU San Pedro de la Paz")
    print("============================================================")
    print("Versión: 2.0 - Con capacidades avanzadas de series temporales")
    print("- Detección de cambios estructurales en patrones epidemiológicos")
    print("- Modelado ARIMA avanzado con identificación automática de parámetros")
    print("- Análisis de ciclos epidémicos y predicción de próximos brotes")
    print("- Estimación del número reproductivo efectivo (Rt)")
    print("============================================================\n")

    # Preguntar al usuario qué tipo de análisis desea realizar
    analysis_type = input("¿Qué tipo de análisis desea realizar? (respiratorio/gastrointestinal/varicela/manopieboca): ").lower()

    while analysis_type not in ['respiratorio', 'gastrointestinal', 'varicela', 'manopieboca']:
        print("Por favor, ingrese 'respiratorio', 'gastrointestinal', 'varicela' o 'manopieboca'.")
        analysis_type = input("¿Qué tipo de análisis desea realizar? (respiratorio/gastrointestinal/varicela/manopieboca): ").lower()

    logging.info(f"Iniciando análisis {analysis_type}")
    print(f"Realizando análisis {analysis_type}...")

    try:
        # Verificar dependencias para análisis avanzado de series temporales
        check_advanced_dependencies()

        # Cargar y preprocesar datos
        df_combined = load_data(data_path, analysis_type)

        # Realizar análisis estadístico avanzado
        print("\nEjecutando análisis estadístico avanzado...")
        print("Esto incluye análisis de series temporales avanzado con detección de cambios estructurales,")
        print("modelado ARIMA, análisis de ciclos epidémicos y métricas de transmisibilidad...")
        analysis_results = perform_advanced_analysis(df_combined, analysis_type)
        print("Análisis estadístico completado.\n")

        # Generar visualizaciones
        print("Generando visualizaciones avanzadas...")
        create_visualizations(df_combined, analysis_type, analysis_results)
        print("Visualizaciones generadas exitosamente.\n")

        # Generar informe
        print("Generando informe epidemiológico...")
        report = generate_and_save_report(analysis_results, analysis_type)
        print("Informe generado exitosamente.\n")

        # Generar análisis con GPT si el usuario lo desea
        run_gpt_analysis(analysis_results, report, analysis_type)

        # Crear visualizaciones interactivas
        print("Generando visualizaciones interactivas...")
        create_interactive_visualizations(df_combined, analysis_type)
        print("Visualizaciones interactivas generadas exitosamente.\n")

        # Imprimir resumen del análisis
        print_analysis_summary(analysis_results)

        # Mostrar mensaje final con instrucciones para visualizar resultados
        print_completion_message(analysis_type)

    except Exception as e:
        handle_error(e, df_combined if 'df_combined' in locals() else None)

def check_advanced_dependencies():
    """
    Verifica la disponibilidad de dependencias avanzadas y muestra advertencias si faltan.
    """
    try:
        import pmdarima
        logging.info("Biblioteca pmdarima disponible para modelado ARIMA avanzado")
    except ImportError:
        logging.warning("Biblioteca pmdarima no disponible, se utilizará SARIMA básico")
        print("\nAdvertencia: La biblioteca pmdarima no está instalada.")
        print("Se utilizará SARIMA básico en lugar de auto-ARIMA con selección automática de parámetros.")
        print("Para obtener resultados óptimos, instale pmdarima: pip install pmdarima\n")
    
    try:
        import ruptures
        logging.info("Biblioteca ruptures disponible para detección de cambios estructurales")
    except ImportError:
        logging.warning("Biblioteca ruptures no disponible, se utilizará detección alternativa")
        print("\nAdvertencia: La biblioteca ruptures no está instalada.")
        print("Se utilizará un método alternativo para detección de cambios estructurales.")
        print("Para obtener resultados óptimos, instale ruptures: pip install ruptures\n")

def load_data(data_path, analysis_type):
    """
    Carga y combina los datos de todos los años.
    """
    logging.info("Cargando y preprocesando datos...")
    print("Cargando y preprocesando datos...")

    data_frames = []
    for year in range(2021, 2026):
        try:
            df = load_and_preprocess_data(
                os.path.join(data_path, f'{year}.txt'),
                year,
                analysis_type
            )
            # Verificar el número correcto de semanas
            max_weeks = get_year_max_weeks(year)
            semanas_unicas = df['Semana Epidemiologica'].nunique()
            if semanas_unicas != max_weeks:
                logging.warning(f"Año {year}: {semanas_unicas} semanas (esperadas: {max_weeks})")

            data_frames.append(df)

        except Exception as e:
            logging.error(f"Error al cargar datos del año {year}: {str(e)}")
            raise

    df_combined = pd.concat(data_frames)
    logging.info(f"Datos combinados: {len(df_combined)} registros")
    print(f"Datos combinados: {len(df_combined)} registros")

    return df_combined

def generate_and_save_report(analysis_results, analysis_type):
    """
    Genera y guarda el informe de análisis.
    """
    logging.info("Generando informe...")
    print("Generando informe...")

    report = generate_report(analysis_results, analysis_type)
    save_report(report, analysis_type)

    return report

def run_gpt_analysis(analysis_results, report_text, analysis_type):
    """
    Ejecuta el análisis GPT si el usuario lo desea.
    """
    run_gpt = input("\n¿Desea realizar el análisis principal con GPT? (s/n): ").lower()

    if run_gpt == 's':
        logging.info("Generando análisis con GPT...")
        print("\nGenerando análisis con GPT...")
        print("El análisis GPT integrará los resultados del análisis avanzado de series temporales...")

        gpt_analysis = generate_gpt_analysis(analysis_results, report_text, analysis_type)
        gpt_output_file = f'gpt_analysis_{analysis_type}.txt'
        save_gpt_analysis(gpt_analysis, gpt_output_file)

        print("\nResumen del análisis de GPT:")
        print(gpt_analysis[:500] + "...")
    else:
        logging.info("Análisis con GPT omitido.")
        print("Análisis con GPT omitido.")

def handle_error(e, df=None):
    """
    Maneja errores durante la ejecución.
    """
    logging.error(f"Error durante el análisis: {str(e)}")
    print(f"\nError durante el análisis: {str(e)}")

    if df is not None:
        log_dataframe_info(df)
    else:
        logging.error("No se pudo crear el DataFrame combinado.")
        print("No se pudo crear el DataFrame combinado. Consulte el archivo de log para más detalles.")

    print("\nSugerencias de solución:")
    print("1. Verifique que los archivos de datos estén en el formato correcto")
    print("2. Asegúrese de que todas las dependencias estén instaladas (pip install -r requirements.txt)")
    print("3. Consulte el archivo de log para más detalles sobre el error")
    print("4. Si es un error de análisis de series temporales, intente instalando las bibliotecas adicionales:")
    print("   - pip install pmdarima ruptures")

    raise

def log_dataframe_info(df):
    """
    Registra información adicional del DataFrame.
    """
    logging.info("\nInformación adicional del DataFrame:")
    logging.info(f"Número total de registros: {len(df)}")
    logging.info(f"Rango de fechas: {df['Fecha Admision'].min()} a {df['Fecha Admision'].max()}")
    logging.info(f"Rango de semanas epidemiológicas: {df['Semana Epidemiologica'].min()} a {df['Semana Epidemiologica'].max()}")
    logging.info(f"Años únicos en el dataset: {df['Año'].unique()}")
    print("\nSe ha registrado información adicional en el archivo de log.")

def print_analysis_summary(analysis_results):
    """
    Imprime un resumen del análisis realizado.
    """
    print("\n=== Resumen del Análisis ===")

    # Resumen de casos por año
    print("\nCasos por año:")
    if 'weekly_cases_by_age_group' in analysis_results:
        for year, data in analysis_results['weekly_cases_by_age_group'].items():
            total_cases = data.drop(columns=['Semana']).sum().sum() # Se excluye la columna semana del conteo
            print(f"  Año {year}: {total_cases:.0f} casos")

    # Resumen de tendencias
    if 'trend' in analysis_results:
        trend = analysis_results['trend']
        trend_dir = "ascendente" if trend > 0 else "descendente" if trend < 0 else "estable"
        print(f"\nTendencia general: {trend:.2f} casos/semana ({trend_dir})")

    # Resumen de brotes
    if 'outbreaks' in analysis_results:
        outbreaks = analysis_results['outbreaks']
        print(f"\nBrotes detectados: {len(outbreaks)}")

    # NUEVO: Resumen de análisis avanzado de series temporales
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    
    if enhanced_ts:
        print("\n=== Resultados del Análisis Avanzado de Series Temporales ===")
        
        # Cambios estructurales
        structural_changes = enhanced_ts.get('structural_changes', {})
        change_points = structural_changes.get('change_points', [])
        if change_points:
            print(f"Cambios estructurales detectados: {len(change_points)}")
        
        # Ciclos epidémicos
        cycles = enhanced_ts.get('epidemic_cycles', {})
        if cycles.get('cycle_detected', False):
            cycle_pattern = cycles.get('cycle_pattern', '')
            pattern_desc = {
                'REGULAR': 'regular',
                'MODERATELY_REGULAR': 'moderadamente regular',
                'IRREGULAR': 'irregular',
                'INSUFFICIENT_DATA': 'con datos insuficientes'
            }
            cycle_desc = pattern_desc.get(cycle_pattern, cycle_pattern).lower()
            
            interval_mean = cycles.get('intervals', {}).get('mean', 0)
            print(f"Ciclo epidémico: Patrón {cycle_desc} con intervalo medio de {interval_mean:.1f} días")
            
            if 'next_outbreak_estimate' in cycles:
                next_date = cycles['next_outbreak_estimate']
                date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
                print(f"Próximo brote estimado: {date_str}")
        
        # Transmisibilidad
        transmissibility = enhanced_ts.get('transmissibility_metrics', {})
        if 'current_rt' in transmissibility:
            rt = transmissibility['current_rt']
            rt_status = "crecimiento epidémico" if rt > 1 else "decrecimiento epidémico"
            print(f"Número reproductivo efectivo (Rt): {rt:.2f} ({rt_status})")
        
        # Pronóstico
        if 'forecast' in enhanced_ts and enhanced_ts['forecast'] is not None:
            forecast = enhanced_ts['forecast']
            if isinstance(forecast, pd.DataFrame) and not forecast.empty:
                last_forecast = forecast['forecast'].iloc[-1]
                forecast_weeks = len(forecast)
                print(f"Pronóstico a {forecast_weeks} semanas: {last_forecast:.1f} casos")

def print_completion_message(analysis_type):
    """
    Imprime mensaje final con instrucciones para visualizar resultados.
    """
    print("\n============================================================")
    print("Análisis epidemiológico completado exitosamente")
    print("============================================================")
    print(f"Tipo de análisis: {analysis_type.capitalize()}")
    print("\nResultados disponibles en:")
    print(f"- Informe: analisis/informe_epidemiologico_{analysis_type}.txt")
    print(f"- Visualizaciones: graphs/graphs{analysis_type}/")
    print(f"- Log: analisis/log_analisis_epidemiologico.txt")
    
    print("\nPara visualizar el dashboard interactivo, ejecute:")
    print("  streamlit run streamlit_dashboard.py")
    
    print("\nGracias por utilizar el sistema de análisis epidemiológico")
    print("============================================================")

if __name__ == "__main__":
    print("Iniciando análisis epidemiológico...")
    main()