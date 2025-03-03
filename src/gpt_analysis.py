import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import logging
import time

# Configurar tu clave API de OpenAI (manteniendo visible como indicado)
API_KEY = "sk-proj-umz5HnyG529nLWLSRVeoT3BlbkFJD8B7imF4k05PnOWkgySG"

# Configurar el cliente de OpenAI con la nueva sintaxis
client = OpenAI(api_key=API_KEY)

# Configurar logging para el módulo
logger = logging.getLogger('gpt_analysis')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('analisis/gpt_analysis.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_year_max_weeks(year):
    """
    Obtiene el número máximo de semanas para un año específico.
    """
    return 53 if year == 2025 else 52

def save_gpt_analysis(analysis_text, output_file):
    """
    Guarda el análisis de GPT en un archivo de texto.
    
    Args:
        analysis_text (str): Texto del análisis generado por GPT
        output_file (str): Nombre del archivo de salida
    """
    os.makedirs('analisis', exist_ok=True)
    file_path = os.path.join('analisis', output_file)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    
    logger.info(f"Análisis de GPT guardado en {file_path}")
    print(f"Análisis de GPT guardado en {file_path}")
    
    # También guardar en formato JSON para facilitar la integración con otras herramientas
    json_file_path = os.path.join('analisis', f"{os.path.splitext(output_file)[0]}.json")
    
    # Convertir el análisis a un formato estructurado para JSON
    analysis_sections = parse_analysis_to_json(analysis_text)
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_sections, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Análisis de GPT en formato JSON guardado en {json_file_path}")
    return file_path

def parse_analysis_to_json(analysis_text):
    """
    Parsea el texto del análisis a un formato estructurado JSON.
    
    Args:
        analysis_text (str): Texto del análisis generado por GPT
        
    Returns:
        dict: Estructura de datos con el análisis parseado
    """
    sections = {}
    current_section = "summary"
    sections[current_section] = []
    
    for line in analysis_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Detectar encabezados de sección (numerados o con #)
        if line.startswith(('#', '##')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
            # Nuevo encabezado de sección detectado
            current_section = line.strip('#. ')
            sections[current_section] = []
        else:
            # Agregar línea a la sección actual
            sections[current_section].append(line)
    
    # Unir las líneas de cada sección
    for section in sections:
        sections[section] = '\n'.join(sections[section])
    
    return sections

def extract_key_insights(analysis_results, analysis_type):
    """
    Extrae los insights clave de los resultados del análisis para crear 
    un contexto más específico para GPT.
    
    Args:
        analysis_results (dict): Resultados completos del análisis
        analysis_type (str): Tipo de análisis epidemiológico
        
    Returns:
        dict: Diccionario con insights relevantes extraídos
    """
    insights = {
        "tipo_analisis": analysis_type,
        "n_total_casos": 0,
        "distribucion_tiempo": {},
        "tendencia": {},
        "brotes": {},
        "establecimiento": {},
        "demografia": {},
        "series_temporales": {}
    }
    
    # Extraer número total de casos
    casos_por_año = {}
    for year, data in analysis_results.get('weekly_cases_by_age_group', {}).items():
        if isinstance(data, pd.DataFrame):
            # Excluir la columna 'Semana' al calcular la suma
            columns_to_sum = [col for col in data.columns if col != 'Semana']
            casos_por_año[year] = data[columns_to_sum].sum().sum()
            insights["n_total_casos"] += int(casos_por_año[year])
    
    insights["distribucion_tiempo"]["casos_por_año"] = casos_por_año
    
    # Extraer información de tendencia
    insights["tendencia"]["coeficiente"] = analysis_results.get('trend')
    insights["tendencia"]["p_valor"] = analysis_results.get('trend_pvalue')
    
    # Extraer información de brotes
    outbreaks = analysis_results.get('outbreaks', pd.DataFrame())
    if isinstance(outbreaks, pd.DataFrame) and not outbreaks.empty:
        insights["brotes"]["n_brotes"] = len(outbreaks)
        insights["brotes"]["fechas"] = [date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date) 
                                        for date in outbreaks.index]
    
    # Análisis avanzado de series temporales
    enhanced_ts = analysis_results.get('enhanced_time_series', {})
    
    # Ciclos epidémicos
    cycles = enhanced_ts.get('epidemic_cycles', {})
    if cycles.get('cycle_detected', False):
        insights["series_temporales"]["ciclos"] = {
            "patron": cycles.get('cycle_pattern'),
            "intervalo_medio": cycles.get('intervals', {}).get('mean'),
            "confianza": cycles.get('confidence')
        }
        
        if 'next_outbreak_estimate' in cycles:
            next_date = cycles['next_outbreak_estimate']
            date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
            insights["series_temporales"]["proximo_brote"] = date_str
    
    # Transmisibilidad
    transmissibility = enhanced_ts.get('transmissibility_metrics', {})
    if 'current_rt' in transmissibility:
        insights["series_temporales"]["rt_actual"] = transmissibility['current_rt']
    
    # Alerta temprana
    early_warning = enhanced_ts.get('early_warning', {})
    if early_warning:
        historical_comp = early_warning.get('current_vs_historical', {})
        if 'status' in historical_comp:
            insights["series_temporales"]["alerta"] = {
                "estado": historical_comp['status'],
                "desviacion": historical_comp.get('percent_deviation')
            }
    
    # Extraer análisis 2025 para la semana 53
    detailed_2025 = analysis_results.get('detailed_2025_analysis', {})
    if detailed_2025:
        week_53 = detailed_2025.get('week_53_analysis', {})
        if week_53:
            insights["semana_53_2025"] = {
                "casos": week_53.get('total_cases'),
                "edad_media": week_53.get('age_distribution', {}).get('mean'),
                "casos_severos": week_53.get('severe_cases')
            }
    
    # Información específica según tipo de análisis
    specific_analysis = {}
    if analysis_type == 'respiratorio':
        for year, data in analysis_results.get('yearly_analysis', {}).items():
            specific_analysis[year] = {
                "influenza": data.get('influenza_cases', 0),
                "neumonia": data.get('pneumonia_cases', 0),
                "covid19": data.get('covid19_cases', 0)
            }
        insights["respiratorio_specific"] = specific_analysis
    
    elif analysis_type == 'gastrointestinal':
        for year, data in analysis_results.get('yearly_analysis', {}).items():
            specific_analysis[year] = {
                "diarrea_infecciosa": data.get('infectious_diarrhea_cases', 0),
                "diarrea_no_infecciosa": data.get('non_infectious_diarrhea_cases', 0),
                "intoxicacion_alimentaria": data.get('food_poisoning_cases', 0)
            }
        insights["gastrointestinal_specific"] = specific_analysis
        
    elif analysis_type in ['varicela', 'manopieboca']:
        for year, data in analysis_results.get('yearly_analysis', {}).items():
            specific_analysis[year] = {
                "casos_menores_5": data.get('under_5_analysis', {}).get('proportion', 0) * data.get('total_cases', 0),
                "severidad": data.get('severity', 0) * 100
            }
        insights[f"{analysis_type}_specific"] = specific_analysis
    
    return insights

def generate_enhanced_prompt(analysis_results, report_text, analysis_type):
    """
    Genera un prompt mejorado y estructurado para GPT basado en los resultados del análisis.
    
    Args:
        analysis_results (dict): Resultados completos del análisis
        report_text (str): Texto del informe generado
        analysis_type (str): Tipo de análisis epidemiológico
        
    Returns:
        str: Prompt estructurado para GPT
    """
    # Extraer insights clave
    insights = extract_key_insights(analysis_results, analysis_type)
    
    # Preparar contexto específico según el tipo de análisis
    analysis_context = ""
    if analysis_type == 'respiratorio':
        analysis_context = """
        Para el análisis respiratorio, considera:
        - La estacionalidad típica de enfermedades respiratorias en el hemisferio sur (invierno: junio-agosto)
        - La importancia de distinguir entre influenza, neumonía, COVID-19 y otras infecciones respiratorias
        - Factores de riesgo específicos como edad (menores de 5 y mayores de 65) y comorbilidades
        - Posibles efectos post-pandemia de COVID-19 en la epidemiología respiratoria
        - La importancia de las vacunas de influenza y COVID-19 en la evolución de casos
        """
    elif analysis_type == 'gastrointestinal':
        analysis_context = """
        Para el análisis gastrointestinal, considera:
        - La estacionalidad típica de gastroenteritis (aumentos en verano y en períodos escolares)
        - Distinguir entre diarreas infecciosas y no infecciosas
        - La probable transmisión comunitaria o intrafamiliar
        - Factores estacionales como temperatura y efecto en alimentos
        - Posibles fuentes comunes de brotes (agua, alimentos, instituciones)
        - Mayores riesgos en poblaciones vulnerables (niños, adultos mayores)
        """
    elif analysis_type == 'varicela':
        analysis_context = """
        Para el análisis de varicela, considera:
        - La alta contagiosidad y comportamiento cíclico de la varicela
        - Impacto de programas de vacunación en la epidemiología
        - Estacionalidad característica (generalmente fin de invierno y primavera)
        - Mayor incidencia en población pediátrica
        - Importancia de prevenir complicaciones en grupos de riesgo
        - Posibles brotes institucionales (escuelas, jardines infantiles)
        """
    elif analysis_type == 'manopieboca':
        analysis_context = """
        Para el análisis de enfermedad mano-pie-boca, considera:
        - La naturaleza altamente contagiosa de esta enfermedad viral
        - Alta incidencia en niños menores de 5 años
        - Estacionalidad típica (mayor incidencia en primavera y verano)
        - Ciclos epidémicos cada 2-3 años en muchas regiones
        - Rol de centros educativos en la transmisión
        - Baja tasa de complicaciones pero alto impacto en ausentismo escolar
        """
    
    # Construir prompt completo
    prompt = f"""
    # Contexto del Análisis Epidemiológico
    
    ## Tipo de Análisis: {analysis_type.upper()}
    
    ## Datos Básicos:
    - Total de casos analizados: {insights['n_total_casos']}
    - Período de tiempo: 2021-2025
    - Área geográfica: San Pedro de la Paz (Chile)
    - Establecimientos: SAR-SAPU San Pedro de la Paz (incluye SAR BOCA SUR, SAR SAN PEDRO, SAPU LOMA COLORADA)
    
    ## Hallazgos Clave del Análisis Estadístico:
    
    ### Tendencia general:
    - Coeficiente de tendencia: {insights['tendencia'].get('coeficiente', 'No disponible')}
    - Significancia estadística (p-valor): {insights['tendencia'].get('p_valor', 'No disponible')}
    
    ### Brotes detectados:
    - Número de brotes: {insights.get('brotes', {}).get('n_brotes', 'No disponible')}
    {f"- Fechas de brotes identificados: {', '.join(insights['brotes'].get('fechas', []))}" if insights.get('brotes', {}).get('fechas') else ''}
    
    ### Análisis Avanzado de Series Temporales:
    {f"- Patrón de ciclo: {insights['series_temporales'].get('ciclos', {}).get('patron', 'No detectado')}" if 'ciclos' in insights.get('series_temporales', {}) else '- No se detectaron ciclos claros'}
    {f"- Próximo brote estimado: {insights['series_temporales'].get('proximo_brote')}" if 'proximo_brote' in insights.get('series_temporales', {}) else ''}
    {f"- Número reproductivo actual (Rt): {insights['series_temporales'].get('rt_actual')}" if 'rt_actual' in insights.get('series_temporales', {}) else ''}
    {f"- Estado de alerta: {insights['series_temporales'].get('alerta', {}).get('estado')} (desviación del promedio histórico: {insights['series_temporales'].get('alerta', {}).get('desviacion')}%)" if 'alerta' in insights.get('series_temporales', {}) else ''}
    
    ### Semana 53 (2025):
    {f"- Total casos: {insights.get('semana_53_2025', {}).get('casos')}, Casos severos: {insights.get('semana_53_2025', {}).get('casos_severos')}" if 'semana_53_2025' in insights else '- No hay información específica sobre la semana 53'}
    
    ## Resumen del informe epidemiológico:
    {report_text[:1500]}  # Primera parte del informe
    
    {analysis_context}
    
    # Solicitud para el Análisis:
    
    Como experto epidemiólogo, realiza un análisis profundo e interpretación de estos hallazgos para {analysis_type}, integrando:
    
    1. RESUMEN EJECUTIVO: Sintetiza los hallazgos más importantes en un párrafo
    
    2. ANÁLISIS DETALLADO DE TENDENCIAS:
       - Interpreta la tendencia general y su significado para la salud pública
       - Analiza la estacionalidad y comportamiento cíclico
       - Evalúa la significancia y características de los brotes detectados
    
    3. ANÁLISIS PREDICTIVO:
       - Interpreta los resultados del modelo ARIMA y su pronóstico
       - Evalúa la confiabilidad del ciclo epidémico detectado y próximo brote estimado
       - Comenta sobre el valor Rt actual y su implicación para la transmisibilidad
    
    4. IMPACTO EN SALUD PÚBLICA:
       - Identifica grupos poblacionales más afectados
       - Evalúa severidad (proporción de hospitalizaciones/derivaciones)
       - Propone estrategias específicas de vigilancia epidemiológica
    
    5. RECOMENDACIONES TÉCNICAS:
       - Medidas de prevención y control basadas en evidencia
       - Optimización de recursos según patrones detectados
       - Propuestas para mejorar la vigilancia epidemiológica
    
    6. PERSPECTIVAS FUTURAS:
       - Proyección a mediano plazo (6-12 meses)
       - Escenarios potenciales según análisis de series temporales
       - Consideraciones sobre preparación para próximos períodos epidémicos
    
    7. OBSERVACIONES ESPECÍFICAS SOBRE LA SEMANA 53 (2025):
       - Impacto de esta semana adicional en el análisis epidemiológico
       - Recomendaciones para manejo estadístico y planificación

    8. LIMITACIONES DEL ANÁLISIS:
       - Identifica posibles sesgos o limitaciones metodológicas
       - Propone mejoras para futuros análisis
    
    Tu análisis debe ser técnicamente riguroso pero comprensible para profesionales de salud, utilizando terminología epidemiológica apropiada. Interpreta los hallazgos en el contexto de la salud pública chilena y las características específicas de {analysis_type}.
    """
    
    return prompt

def call_gpt_with_retry(messages, max_retries=3, initial_wait=2):
    """
    Llama a la API de GPT con reintentos en caso de fallos.
    
    Args:
        messages (list): Lista de mensajes para la API
        max_retries (int): Número máximo de reintentos
        initial_wait (int): Tiempo de espera inicial entre reintentos (se duplica con cada intento)
        
    Returns:
        str: Respuesta de GPT o mensaje de error
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Intento {attempt + 1} de llamada a GPT")
            
            # Usar la nueva estructura de llamada a la API para modelo o1
            response = client.chat.completions.create(
                model="o1",  # Usar el modelo o1 como solicitado
                messages=messages,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            logger.error(f"Error en llamada a GPT (intento {attempt + 1}): {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Esperando {wait_time} segundos antes de reintentar...")
                time.sleep(wait_time)
            else:
                return f"Error al generar el análisis de GPT después de {max_retries} intentos: {str(e)}"

def generate_gpt_analysis(analysis_results, report_text, analysis_type):
    """
    Genera un análisis utilizando GPT basado en los resultados del análisis epidemiológico.
    
    Args:
        analysis_results (dict): Resultados del análisis estadístico
        report_text (str): Texto del informe generado
        analysis_type (str): Tipo de análisis ('respiratorio', 'gastrointestinal', 'varicela', 'manopieboca')
        
    Returns:
        str: Análisis generado por GPT
    """
    logger.info(f"Iniciando generación de análisis GPT para {analysis_type}")
    
    # Generar prompt mejorado y estructurado
    prompt = generate_enhanced_prompt(analysis_results, report_text, analysis_type)
    
    # Construir mensajes para la API con el nuevo formato para o1
    messages = [
        {"role": "developer", "content": f"Eres un epidemiólogo senior especializado en vigilancia epidemiológica y análisis de brotes de {analysis_type}, con amplia experiencia en salud pública en Chile y Latinoamérica."},
        {"role": "user", "content": prompt}
    ]
    
    # Llamar a GPT con reintentos
    response = call_gpt_with_retry(messages)
    
    logger.info(f"Análisis GPT para {analysis_type} generado con éxito")
    
    # Añadir un header al análisis con metadatos
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    analysis_with_header = f"""
    # ANÁLISIS EPIDEMIOLÓGICO GENERADO POR IA - {analysis_type.upper()}
    
    Fecha de generación: {timestamp}
    Tipo de análisis: {analysis_type}
    Período analizado: 2021-2025
    Establecimientos: SAR-SAPU San Pedro de la Paz
    
    ----------------------------------------
    
    {response}
    """
    
    return analysis_with_header

def generate_comparative_analysis(analysis_results, previous_results=None):
    """
    Genera un análisis comparativo entre resultados actuales y anteriores.
    Útil para comparación de diferentes períodos o antes/después de intervenciones.
    
    Args:
        analysis_results (dict): Resultados actuales
        previous_results (dict, optional): Resultados de un período anterior
        
    Returns:
        str: Análisis comparativo generado por GPT
    """
    if previous_results is None:
        logger.warning("No se proporcionaron resultados previos para comparación")
        return "No se pudo generar análisis comparativo (resultados previos no disponibles)"
    
    # Implementar lógica para extraer diferencias clave entre los dos conjuntos de resultados
    # y generar un prompt específico para análisis comparativo
    
    # Este es un ejemplo simplificado
    prompt = f"""
    Realiza un análisis comparativo detallado entre dos períodos de vigilancia epidemiológica,
    identificando cambios significativos en tendencias, estacionalidad, severidad y distribución poblacional.
    
    Período actual: {analysis_results.get('date_range', 'No especificado')}
    Período anterior: {previous_results.get('date_range', 'No especificado')}
    
    Cambios en tendencia:
    - Tendencia actual: {analysis_results.get('trend', 'No disponible')}
    - Tendencia anterior: {previous_results.get('trend', 'No disponible')}
    
    Cambios en brotes detectados:
    - Brotes actuales: {len(analysis_results.get('outbreaks', []))}
    - Brotes anteriores: {len(previous_results.get('outbreaks', []))}
    
    Por favor, proporciona:
    1. Evaluación de la dirección y magnitud de los cambios
    2. Posibles factores contribuyentes a los cambios observados
    3. Implicaciones para la vigilancia epidemiológica y políticas de salud pública
    4. Recomendaciones específicas basadas en la comparación
    """
    
    # Llamar a GPT para generar el análisis comparativo con el nuevo formato para o1
    messages = [
        {"role": "developer", "content": "Eres un epidemiólogo especializado en análisis comparativo de datos de vigilancia epidemiológica"},
        {"role": "user", "content": prompt}
    ]
    
    logger.info("Generando análisis comparativo con GPT")
    response = call_gpt_with_retry(messages)
    
    return response

if __name__ == "__main__":
    print("Este módulo está diseñado para ser importado y utilizado en main.py")
    print("Para realizar el análisis de GPT, ejecute main.py")