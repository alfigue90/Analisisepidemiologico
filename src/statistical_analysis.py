import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Nuevas importaciones para análisis avanzado de series temporales
try:
    import pmdarima as pm
    from pmdarima.arima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima no está instalado. Se utilizará SARIMAX básico en su lugar.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures no está instalado. No se realizará detección de cambios estructurales.")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_year_max_weeks(year):
    """
    Retorna el número máximo de semanas para un año específico.

    Args:
        year (int): Año a consultar

    Returns:
        int: Número máximo de semanas (52 o 53)
    """
    return 53 if year == 2025 else 52

def analyze_weekly_cases_by_age_group(df):
    """
    Analiza las atenciones semanales por rango etario para cada año.

    Args:
        df (pandas.DataFrame): DataFrame con los datos

    Returns:
        dict: Diccionario con análisis por año
    """
    age_groups = ['Menor de 1 A', '1 a 4 A', '5 a 14 A', '15 a 64 A', '65 y más A']
    results = {}

    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        weekly_data = []
        max_weeks = get_year_max_weeks(year)

        for week in range(1, max_weeks + 1):
            week_data = df_year[df_year['Semana Epidemiologica'] == week]
            week_counts = week_data['Grupo_Edad'].value_counts()
            week_dict = {'Semana': week}
            for group in age_groups:
                week_dict[group] = week_counts.get(group, 0)
            weekly_data.append(week_dict)

        results[year] = pd.DataFrame(weekly_data)

    return results

def prepare_weekly_data(df):
    """
    Prepara los datos semanales para análisis, manejando años con diferente número de semanas.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos epidemiológicos
        
    Returns:
        pandas.DataFrame: DataFrame con datos semanales preparados e indexados por fecha
    """
    weekly_data = []

    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        for week in range(1, max_weeks + 1):
            week_data = df_year[df_year['Semana Epidemiologica'] == week]
            week_dict = {
                'Año': year,
                'Semana Epidemiologica': week,
                'Casos': len(week_data),
                'Edad_Promedio': week_data['Edad_Anios'].mean() if not week_data.empty else np.nan
            }
            weekly_data.append(week_dict)

    df_weekly = pd.DataFrame(weekly_data)
    # Convertir 'Semana Epidemiologica' a entero
    df_weekly['Semana Epidemiologica'] = df_weekly['Semana Epidemiologica'].astype(int)
    df_weekly['fecha'] = df_weekly.apply(lambda row:
                                         datetime.strptime(f"{int(row['Año'])}-W{int(row['Semana Epidemiologica'])}-1", "%Y-W%W-%w"), axis=1)

    return df_weekly.sort_values('fecha').set_index('fecha')

def analyze_trend(df_weekly):
    """
    Analiza la tendencia considerando años con diferente número de semanas.
    
    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales
        
    Returns:
        dict: Resultados del análisis de tendencia
    """
    # Crear variable tiempo normalizada para comparabilidad entre años
    df_weekly['tiempo_normalizado'] = (df_weekly.index - df_weekly.index.min()).days / 7

    X = sm.add_constant(df_weekly['tiempo_normalizado'])
    model = sm.OLS(df_weekly['Casos'], X, missing='drop')
    results_trend = model.fit()

    return {
        'trend': results_trend.params.iloc[1],
        'trend_pvalue': results_trend.pvalues.iloc[1],
        'trend_confidence_interval': results_trend.conf_int().iloc[1].tolist()
    }

def analyze_derivations(df):
    """
    Analiza las derivaciones de pacientes a otros centros de salud.

    Args:
        df (pandas.DataFrame): DataFrame con los datos de pacientes

    Returns:
        dict: Diccionario con resultados del análisis de derivaciones
    """
    derivation_results = {}

    # Asegurar que la columna Destino no tenga valores nulos
    df = df.copy()
    df['Destino'] = df['Destino'].fillna('NO ESPECIFICADO')

    # Análisis general de derivaciones
    derivation_counts = df['Destino'].value_counts()
    total_cases = len(df)
    derivation_proportions = derivation_counts / total_cases * 100
    derivation_proportions = {k: round(v, 2) for k, v in derivation_proportions.items()}

    derivation_results['general'] = {
        'counts': derivation_counts.to_dict(),
        'proportions': derivation_proportions,
        'total_cases': total_cases
    }

    # Análisis por año
    yearly_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        year_counts = df_year['Destino'].value_counts()
        year_total = len(df_year)
        year_proportions = year_counts / year_total * 100 if year_total > 0 else year_counts * 0
        year_proportions = {k: round(v, 2) for k, v in year_proportions.items()}

        yearly_analysis[year] = {
            'counts': year_counts.to_dict(),
            'proportions': year_proportions,
            'total_cases': year_total
        }

    derivation_results['yearly'] = yearly_analysis

    # Análisis por grupo de edad
    age_analysis = {}
    for grupo in df['Grupo_Edad'].unique():
        df_grupo = df[df['Grupo_Edad'] == grupo]
        grupo_counts = df_grupo['Destino'].value_counts()
        grupo_total = len(df_grupo)
        grupo_proportions = grupo_counts / grupo_total * 100 if grupo_total > 0 else grupo_counts * 0
        grupo_proportions = {k: round(v, 2) for k, v in grupo_proportions.items()}

        age_analysis[grupo] = {
            'counts': grupo_counts.to_dict(),
            'proportions': grupo_proportions,
            'total_cases': grupo_total
        }

    derivation_results['by_age_group'] = age_analysis

    # Análisis por establecimiento
    establishment_analysis = {}
    for estab in df['Estableciemiento'].unique():
        df_estab = df[df['Estableciemiento'] == estab]
        estab_counts = df_estab['Destino'].value_counts()
        estab_total = len(df_estab)
        estab_proportions = estab_counts / estab_total * 100 if estab_total > 0 else estab_counts * 0
        estab_proportions = {k: round(v, 2) for k, v in estab_proportions.items()}

        establishment_analysis[estab] = {
            'counts': estab_counts.to_dict(),
            'proportions': estab_proportions,
            'total_cases': estab_total
        }

    derivation_results['by_establishment'] = establishment_analysis

    # Análisis temporal (por mes)
    monthly_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        monthly_counts = {}

        for month in range(1, 13):
            df_month = df_year[df_year['Mes'] == month]
            month_total = len(df_month)
            if month_total > 0:
                month_counts = df_month['Destino'].value_counts()
                month_proportions = month_counts / month_total * 100
                month_proportions = {k: round(v, 2) for k, v in month_proportions.items()}

                monthly_counts[month] = {
                    'counts': month_counts.to_dict(),
                    'proportions': month_proportions,
                    'total_cases': month_total
                }

        monthly_analysis[year] = monthly_counts

    derivation_results['monthly'] = monthly_analysis

    # Análisis de severidad
    total_derivations = len(df[df['Destino'] != 'DOMICILIO'])
    severity_analysis = {
        'total_derivations': total_derivations,
        'derivation_rate': round(total_derivations / total_cases * 100, 2) if total_cases > 0 else 0,
        'destinations': df[df['Destino'] != 'DOMICILIO']['Destino'].value_counts().to_dict()
    }

    derivation_results['severity'] = severity_analysis

    # Análisis estadístico básico
    if total_cases > 0:
        derivation_stats = {
            'domicilio_rate': round(len(df[df['Destino'] == 'DOMICILIO']) / total_cases * 100, 2),
            'hospitalization_rate': round(len(df[df['Destino'].str.contains('HOSPITAL', na=False)]) / total_cases * 100, 2),
            'referral_rate': round(len(df[df['Destino'].str.contains('DERIVADO|REFERIDO', na=False, regex=True)]) / total_cases * 100, 2)
        }

        derivation_results['stats'] = derivation_stats

    return derivation_results

def analyze_proportions(df):
    """
    Realiza análisis de proporciones para diferentes categorías usando métodos robustos.

    Args:
        df (pandas.DataFrame): DataFrame con los datos de pacientes

    Returns:
        dict: Diccionario con resultados del análisis de proporciones
    """
    from scipy import stats
    import numpy as np

    proportion_results = {}

    def calculate_basic_stats(data, total):
        """
        Calcula estadísticas básicas de proporción con intervalos de confianza.
        """
        if total == 0:
            return {
                'count': 0,
                'proportion': 0,
                'percentage': 0,
                'ci_lower': 0,
                'ci_upper': 0
            }

        proportion = data / total
        percentage = proportion * 100

        # Calculando intervalo de confianza usando método normal aproximado
        z = 1.96  # Para 95% de confianza
        if total > 0:
            std_err = np.sqrt((proportion * (1 - proportion)) / total)
            ci_lower = max(0, proportion - z * std_err)
            ci_upper = min(1, proportion + z * std_err)
        else:
            ci_lower, ci_upper = 0, 0

        return {
            'count': int(data),
            'proportion': round(proportion, 4),
            'percentage': round(percentage, 2),
            'ci_lower': round(ci_lower * 100, 2),
            'ci_upper': round(ci_upper * 100, 2)
        }

    total_cases = len(df)

    # Análisis por género
    gender_counts = df['Sexo'].value_counts()
    proportion_results['gender'] = {
        gender: calculate_basic_stats(count, total_cases)
        for gender, count in gender_counts.items()
    }

    # Análisis por grupo de edad
    age_counts = df['Grupo_Edad'].value_counts()
    proportion_results['age_groups'] = {
        age: calculate_basic_stats(count, total_cases)
        for age, count in age_counts.items()
    }

    # Análisis por establecimiento
    establishment_counts = df['Estableciemiento'].value_counts()
    proportion_results['establishments'] = {
        est: calculate_basic_stats(count, total_cases)
        for est, count in establishment_counts.items()
    }

    # Análisis por destino
    destination_counts = df['Destino'].value_counts()
    proportion_results['destinations'] = {
        dest: calculate_basic_stats(count, total_cases)
        for dest, count in destination_counts.items()
    }

    # Análisis temporal por año y mes
    temporal_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        year_cases = len(df_year)

        monthly_counts = df_year['Mes'].value_counts()
        temporal_analysis[year] = {
            month: calculate_basic_stats(monthly_counts.get(month, 0), year_cases)
            for month in range(1, 13)
        }
    proportion_results['temporal'] = temporal_analysis

    # Análisis de severidad general
    severe_cases = len(df[df['Destino'] != 'DOMICILIO'])
    proportion_results['severity'] = calculate_basic_stats(severe_cases, total_cases)

    # Top 10 diagnósticos
    diagnosis_counts = df['Diagnostico Principal'].value_counts().head(10)
    proportion_results['top_diagnoses'] = {
        diag: calculate_basic_stats(count, total_cases)
        for diag, count in diagnosis_counts.items()
    }

    # Análisis por país de origen
    origin_counts = df['Pais_Origen'].value_counts()
    proportion_results['origin_countries'] = {
        country: calculate_basic_stats(count, total_cases)
        for country, count in origin_counts.items()
    }

    # Análisis de chi-cuadrado simplificado
    if total_cases > 0:
        try:
            # Género vs Grupo de edad
            contingency_gender_age = pd.crosstab(df['Sexo'], df['Grupo_Edad'])
            chi2, p_value = stats.chi2_contingency(contingency_gender_age)[:2]
            proportion_results['chi_square_tests'] = {
                'gender_vs_age': {
                    'chi2_statistic': round(chi2, 4),
                    'p_value': round(p_value, 4)
                }
            }
        except Exception as e:
            proportion_results['chi_square_tests'] = {
                'error': str(e)
            }

    return proportion_results

def analyze_correlations(df):
    """
    Analiza las correlaciones entre diferentes variables numéricas.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos epidemiológicos
        
    Returns:
        dict: Resultados del análisis de correlaciones
    """
    correlation_results = {}

    # Variables numéricas para correlación
    numeric_columns = ['Edad_Anios', 'Semana Epidemiologica']

    # Matriz de correlación básica
    corr_matrix = df[numeric_columns].corr()
    correlation_results['basic'] = {
        col1: {col2: round(corr_matrix.loc[col1, col2], 4)
               for col2 in corr_matrix.columns}
        for col1 in corr_matrix.index
    }

    # Correlación entre edad y severidad
    df['is_severe'] = (df['Destino'] != 'DOMICILIO').astype(int)
    correlation_results['age_severity'] = round(
        df['Edad_Anios'].corr(df['is_severe']), 4
    )

    # Correlaciones por año
    yearly_correlations = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        if len(df_year) > 0:
            year_corr = df_year[numeric_columns].corr()
            yearly_correlations[year] = {
                col1: {col2: round(year_corr.loc[col1, col2], 4)
                       for col2 in year_corr.columns}
                for col1 in year_corr.index
            }
    correlation_results['yearly'] = yearly_correlations

    return correlation_results

def analyze_seasonality(df_weekly):
    """
    Analiza la estacionalidad con manejo adaptativo del período.
    
    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales
        
    Returns:
        object: Resultado de la descomposición estacional
    """
    try:
        # Determinar el período base en 52 semanas
        base_period = 52

        # Ajustar el período si hay datos suficientes
        n_years = len(df_weekly['Año'].unique())
        if n_years >= 2:
            # Usar datos de años completos para detectar el período
            complete_years_data = df_weekly.groupby('Año').filter(lambda x: len(x) >= 52)
            if not complete_years_data.empty:
                return seasonal_decompose(complete_years_data['Casos'],
                                          period=base_period,
                                          model='additive',
                                          extrapolate_trend='freq')

        return None
    except Exception as e:
        print(f"Error en análisis de estacionalidad: {str(e)}")
        return None

def fit_sarima_model(df_weekly):
    """
    Ajusta un modelo SARIMA con manejo adaptativo de la estacionalidad.
    
    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales
        
    Returns:
        object: Modelo SARIMA ajustado
    """
    try:
        # Usar 52 semanas como período base para estacionalidad
        model_sarima = SARIMAX(df_weekly['Casos'],
                              order=(1, 1, 1),
                              seasonal_order=(1, 1, 1, 52))
        return model_sarima.fit(disp=False)
    except Exception as e:
        print(f"Error al ajustar modelo SARIMA: {str(e)}")
        return None

def detect_outliers(df_weekly):
    """
    Detecta outliers en los datos semanales, considerando la estructura del año.

    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales

    Returns:
        pandas.DataFrame: DataFrame con los outliers detectados
    """
    outliers_by_year = []

    for year in df_weekly.index.year.unique():
        df_year = df_weekly[df_weekly.index.year == year]

        # Ajustar la contaminación según el número de semanas
        n_weeks = get_year_max_weeks(year)
        contamination = 0.1 * (n_weeks / 52)  # Ajuste proporcional

        clf = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = clf.fit_predict(df_year[['Casos']])
        outliers_year = df_year[outlier_labels == -1]
        outliers_by_year.append(outliers_year)

    if outliers_by_year:
        return pd.concat(outliers_by_year)
    else:
        return pd.DataFrame()

def detect_outbreaks(df_weekly):
    """
    Detecta brotes en los datos semanales, adaptado para diferentes longitudes de año.

    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales

    Returns:
        pandas.DataFrame: DataFrame con los brotes detectados
    """
    outbreaks_by_year = []

    for year in df_weekly.index.year.unique():
        df_year = df_weekly[df_weekly.index.year == year]

        # Calcular umbral específico para el año
        threshold = df_year['Casos'].mean() + 2 * df_year['Casos'].std()

        # Encontrar picos
        peaks, _ = find_peaks(df_year['Casos'].values,
                             height=threshold,
                             distance=2)  # Mínimo 2 semanas entre picos

        outbreaks_year = df_year.iloc[peaks]
        outbreaks_by_year.append(outbreaks_year)

    if outbreaks_by_year:
        return pd.concat(outbreaks_by_year)
    else:
        return pd.DataFrame()

def analyze_autocorrelation(df_weekly):
    """
    Analiza la autocorrelación en los datos, adaptado para diferentes longitudes de año.

    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales

    Returns:
        dict: Diccionario con resultados de autocorrelación
    """
    # Calcular el máximo de lags basado en la estructura de los datos
    max_lags = min(len(df_weekly) // 2, 52)  # Usar 52 como máximo estándar

    try:
        acf_values = acf(df_weekly['Casos'], nlags=max_lags)
        pacf_values = pacf(df_weekly['Casos'], nlags=max_lags)

        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'max_lags': max_lags
        }
    except Exception as e:
        print(f"Error en análisis de autocorrelación: {str(e)}")
        return {
            'acf': [],
            'pacf': [],
            'max_lags': max_lags
        }

def analyze_group(df, group):
    """
    Realiza análisis por grupo específico, con manejo mejorado de datos temporales.

    Args:
        df (pandas.DataFrame): DataFrame con los datos
        group (str): Nombre del grupo a analizar

    Returns:
        pandas.DataFrame: DataFrame con el análisis del grupo
    """
    # Análisis básico por grupo
    group_analysis = df.groupby([group, 'Año']).agg({
        'CIE10 DP': 'count',
        'Edad_Anios': ['mean', 'std']
    }).reset_index()

    # Reorganizar columnas
    group_analysis.columns = [group, 'Año', 'Casos', 'Edad_Promedio', 'Edad_Std']

    # Calcular tasas por semana
    for year in group_analysis['Año'].unique():
        max_weeks = get_year_max_weeks(year)
        mask = group_analysis['Año'] == year
        group_analysis.loc[mask, 'Tasa_Semanal'] = group_analysis.loc[mask, 'Casos'] / max_weeks

    return group_analysis

def analyze_diagnoses(df):
    """
    Analiza los diagnósticos, con agrupación temporal mejorada.

    Args:
        df (pandas.DataFrame): DataFrame con los datos

    Returns:
        dict: Diccionario con análisis de diagnósticos
    """
    # Análisis general
    top_10_general = df['Diagnostico Principal'].value_counts().head(10)

    # Análisis por año
    diagnoses_by_year = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        diagnoses_by_year[year] = df_year['Diagnostico Principal'].value_counts().head(10)

    return {
        'top_10_general': top_10_general,
        'by_year': diagnoses_by_year
    }

def analyze_annual_trends(df):
    """
    Analiza tendencias anuales, considerando diferentes longitudes de año.

    Args:
        df (pandas.DataFrame): DataFrame con los datos

    Returns:
        dict: Diccionario con tendencias anuales
    """
    annual_trends = {}

    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        weekly_cases = df_year.groupby('Semana Epidemiologica')['CIE10 DP'].count()

        # Normalizar semanas para comparabilidad
        X = sm.add_constant(np.arange(len(weekly_cases)) / max_weeks)
        model = sm.OLS(weekly_cases, X)
        results = model.fit()

        annual_trends[year] = {
            'trend': results.params[1],
            'p_value': results.pvalues[1],
            'r_squared': results.rsquared,
            'max_weeks': max_weeks,
            'total_cases': len(df_year)
        }

    return annual_trends

def analyze_weekly_change(df):
    """
    Analiza el cambio semanal, adaptado para diferentes longitudes de año.

    Args:
        df (pandas.DataFrame): DataFrame con los datos

    Returns:
        dict: Diccionario con análisis de cambio semanal
    """
    weekly_change = {}

    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        # Agrupar por semana y contar los casos
        weekly_cases = df_year.groupby('Semana Epidemiologica')['CIE10 DP'].count()

        # Calcular cambios porcentuales respecto a la semana anterior
        changes = weekly_cases.pct_change() * 100

        # Almacenar resultados para el año actual
        weekly_change[year] = changes.to_dict()

    return weekly_change

def analyze_respiratory_diseases(df):
    """
    Análisis específico para enfermedades respiratorias.

    Args:
        df (pandas.DataFrame): DataFrame con datos respiratorios (ya filtrados por data_processing.py)

    Returns:
        dict: Diccionario con resultados del análisis
    """
    respiratory_results = {}

    # Análisis de tipos de infección
    respiratory_results['infection_types'] = df['CIE10 DP'].str[:3].value_counts()

    # Análisis por año considerando semanas epidemiológicas
    yearly_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year].copy() # Copia para evitar SettingWithCopyWarning
        max_weeks = get_year_max_weeks(year)

        # Usamos na=False en todas las llamadas a str.startswith()
        year_data = {
            'total_cases': len(df_year),
            'cases_per_week': len(df_year) / max_weeks,
            'influenza_cases': df_year[df_year['CIE10 DP'].str.startswith(('J09', 'J10', 'J11'), na=False)].shape[0],
            'pneumonia_cases': df_year[df_year['CIE10 DP'].str.startswith(('J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18'), na=False)].shape[0],
            'covid19_cases': df_year[df_year['CIE10 DP'].str.startswith('U07', na=False)].shape[0],
            'other_respiratory_cases': df_year[~df_year['CIE10 DP'].str.startswith(('J09', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'U07'), na=False)].shape[0]
        }

        # Análisis de severidad
        year_data['severity'] = df_year[df_year['Destino'] != 'DOMICILIO'].shape[0] / len(df_year)

        # Análisis estacional
        seasonal_pattern = df_year.groupby('Mes')['CIE10 DP'].count()
        year_data['seasonal_pattern'] = seasonal_pattern.to_dict()

        yearly_analysis[year] = year_data

    respiratory_results['yearly_analysis'] = yearly_analysis

    # Análisis de comorbilidades y complicaciones
    respiratory_results['severity_analysis'] = {
        'hospitalization_rate': df[df['Destino'] != 'DOMICILIO'].shape[0] / len(df),
        'age_severity_correlation': df.groupby('Grupo_Edad')['Destino'].apply(
            lambda x: (x != 'DOMICILIO').mean()
        ).to_dict()
    }

    return respiratory_results

def analyze_cases_by_comuna_for_disease(df):
    """
    Función auxiliar para analizar casos por comuna para una enfermedad específica.
    
    Args:
        df (pandas.DataFrame): DataFrame filtrado para la enfermedad específica
        
    Returns:
        dict: Diccionario con análisis por comuna a nivel semanal, mensual y anual
    """
    comuna_analysis = {}
    
    # Análisis semanal
    weekly_comuna = df.groupby(['Año', 'Semana Epidemiologica', 'Comuna'])['CIE10 DP'].count().reset_index(name='Casos')
    weekly_distribution = {}
    for year in df['Año'].unique():
        weekly_distribution[year] = weekly_comuna[weekly_comuna['Año'] == year].pivot(
            index='Semana Epidemiologica',
            columns='Comuna',
            values='Casos'
        ).fillna(0).to_dict()
    
    # Análisis mensual
    monthly_comuna = df.groupby(['Año', 'Mes', 'Comuna'])['CIE10 DP'].count().reset_index(name='Casos')
    monthly_distribution = {}
    for year in df['Año'].unique():
        monthly_distribution[year] = monthly_comuna[monthly_comuna['Año'] == year].pivot(
            index='Mes',
            columns='Comuna',
            values='Casos'
        ).fillna(0).to_dict()
    
    # Análisis anual
    annual_comuna = df.groupby(['Año', 'Comuna'])['CIE10 DP'].count().reset_index(name='Casos')
    annual_distribution = annual_comuna.pivot(
        index='Año',
        columns='Comuna',
        values='Casos'
    ).fillna(0).to_dict()
    
    return {
        'weekly_distribution': weekly_distribution,
        'monthly_distribution': monthly_distribution,
        'annual_distribution': annual_distribution
    }

def analyze_gastrointestinal_diseases(df):
    """
    Análisis específico para enfermedades gastrointestinales.

    Args:
        df (pandas.DataFrame): DataFrame con datos gastrointestinales

    Returns:
        dict: Diccionario con resultados del análisis
    """
    gastrointestinal_results = {}

    # Asegurarse de que CIE10 DP no tenga valores nulos
    df = df.copy()
    df['CIE10 DP'] = df['CIE10 DP'].fillna('')

    # Análisis de tipos de infección
    gastrointestinal_results['infection_types'] = df['CIE10 DP'].str[:3].value_counts()

    # Análisis por año considerando semanas epidemiológicas
    yearly_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        # Crear máscaras de filtrado evitando problemas con NaN
        mask_infectious = df_year['CIE10 DP'].str.startswith(('A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'), na=False)
        mask_food_poisoning = df_year['CIE10 DP'].str.startswith('A05', na=False)
        mask_viral_gastro = df_year['CIE10 DP'].str.startswith('A08', na=False)

        year_data = {
            'total_cases': len(df_year),
            'cases_per_week': len(df_year) / max_weeks,
            'infectious_diarrhea_cases': mask_infectious.sum(),
            'non_infectious_diarrhea_cases': (~mask_infectious).sum(),
            'food_poisoning_cases': mask_food_poisoning.sum(),
            'viral_gastroenteritis_cases': mask_viral_gastro.sum()
        }

        # Análisis de severidad
        mask_no_domicilio = df_year['Destino'] != 'DOMICILIO'
        year_data['severity'] = mask_no_domicilio.sum() / len(df_year) if len(df_year) > 0 else 0

        # Análisis estacional
        seasonal_pattern = df_year.groupby('Mes')['CIE10 DP'].count()
        year_data['seasonal_pattern'] = seasonal_pattern.to_dict()

        yearly_analysis[year] = year_data

    gastrointestinal_results['yearly_analysis'] = yearly_analysis

    # Análisis de factores de riesgo
    risk_analysis = {}

    # Análisis por grupo de edad
    age_group_risk = df.groupby('Grupo_Edad').apply(
        lambda x: (x['Destino'] != 'DOMICILIO').mean() if len(x) > 0 else 0
    ).to_dict()
    risk_analysis['age_group_risk'] = age_group_risk

    # Análisis por tipo de infección
    severity_by_type = df.groupby('CIE10 DP').apply(
        lambda x: (x['Destino'] != 'DOMICILIO').mean() if len(x) > 0 else 0
    ).to_dict()
    risk_analysis['severity_by_type'] = severity_by_type

    gastrointestinal_results['risk_analysis'] = risk_analysis
    
    # Añadir análisis por comuna
    gastrointestinal_results['comuna_analysis'] = analyze_cases_by_comuna_for_disease(df)

    return gastrointestinal_results

def analyze_varicela_diseases(df):
    """
    Análisis específico para casos de varicela.

    Args:
        df (pandas.DataFrame): DataFrame con datos de varicela

    Returns:
        dict: Diccionario con resultados del análisis
    """
    varicela_results = {}

    # Análisis por año considerando semanas epidemiológicas
    yearly_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        year_data = {
            'total_cases': len(df_year),
            'cases_per_week': len(df_year) / max_weeks,
            'age_distribution': df_year['Edad_Anios'].describe().to_dict(),
            'gender_distribution': df_year['Sexo'].value_counts(normalize=True).to_dict()
        }

        # Análisis de severidad
        year_data['severity'] = df_year[df_year['Destino'] != 'DOMICILIO'].shape[0] / len(df_year)

        # Análisis estacional
        monthly_pattern = df_year.groupby('Mes')['CIE10 DP'].count()
        year_data['monthly_pattern'] = monthly_pattern.to_dict()

        # Análisis por grupo etario
        year_data['age_group_distribution'] = df_year['Grupo_Edad'].value_counts(normalize=True).to_dict()

        yearly_analysis[year] = year_data

    varicela_results['yearly_analysis'] = yearly_analysis

    # Análisis de complicaciones
    varicela_results['complications_analysis'] = {
        'hospitalization_rate': df[df['Destino'] != 'DOMICILIO'].shape[0] / len(df),
        'age_severity_correlation': df.groupby('Grupo_Edad')['Destino'].apply(
            lambda x: (x != 'DOMICILIO').mean()
        ).to_dict()
    }

    return varicela_results

def analyze_manopieboca_diseases(df):
    """
    Análisis específico para casos de enfermedad mano-pie-boca.

    Args:
        df (pandas.DataFrame): DataFrame con datos de mano-pie-boca

    Returns:
        dict: Diccionario con resultados del análisis
    """
    manopieboca_results = {}

    # Análisis por año considerando semanas epidemiológicas
    yearly_analysis = {}
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        year_data = {
            'total_cases': len(df_year),
            'cases_per_week': len(df_year) / max_weeks,
            'age_distribution': df_year['Edad_Anios'].describe().to_dict(),
            'gender_distribution': df_year['Sexo'].value_counts(normalize=True).to_dict()
        }

        # Análisis de severidad
        year_data['severity'] = df_year[df_year['Destino'] != 'DOMICILIO'].shape[0] / len(df_year)

        # Análisis estacional
        monthly_pattern = df_year.groupby('Mes')['CIE10 DP'].count()
        year_data['monthly_pattern'] = monthly_pattern.to_dict()

        # Análisis específico para menores de 5 años
        df_under_5 = df_year[df_year['Edad_Anios'] < 5]
        year_data['under_5_analysis'] = {
            'proportion': len(df_under_5) / len(df_year),
            'severity_rate': df_under_5[df_under_5['Destino'] != 'DOMICILIO'].shape[0] / len(df_under_5) if len(df_under_5) > 0 else 0
        }

        yearly_analysis[year] = year_data

    manopieboca_results['yearly_analysis'] = yearly_analysis

    # Análisis de factores de riesgo
    manopieboca_results['risk_analysis'] = {
        'age_group_risk': df.groupby('Grupo_Edad')['Destino'].apply(
            lambda x: (x != 'DOMICILIO').mean()
        ).to_dict(),
        'severity_by_age': df.groupby('Edad_Anios')['Destino'].apply(
            lambda x: (x != 'DOMICILIO').mean()
        ).to_dict()
    }

    return manopieboca_results

def analyze_by_establishment(df, analysis_type):
    """
    Realiza análisis por establecimiento de salud.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos epidemiológicos
        analysis_type (str): Tipo de análisis epidemiológico
        
    Returns:
        dict: Resultados del análisis por establecimiento
    """
    establishment_results = {}

    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        establishment_data_year = {}
        for estab in df_year['Estableciemiento'].unique():
            df_estab = df_year[df_year['Estableciemiento'] == estab]

            weekly_cases = df_estab.groupby('Semana Epidemiologica').size().to_dict()
            establishment_data_year[estab] = {
                'total_cases': len(df_estab),
                'avg_age': round(df_estab['Edad_Anios'].mean(), 2),
                'severity_rate': round(len(df_estab[df_estab['Destino'] != 'DOMICILIO']) / len(df_estab) * 100, 2) if len(df_estab) > 0 else 0,
                'weekly_cases': {int(week): cases for week, cases in weekly_cases.items()}
            }

        establishment_results[year] = establishment_data_year

    return establishment_results

# ----- NUEVAS FUNCIONES PARA ANÁLISIS AVANZADO DE SERIES TEMPORALES -----

def detect_structural_changes(time_series, min_size=5):
    """
    Detecta cambios estructurales (puntos de cambio) en una serie temporal epidemiológica.
    
    Args:
        time_series (pd.Series): Serie temporal con casos epidemiológicos
        min_size (int): Tamaño mínimo de segmento entre puntos de cambio
        
    Returns:
        dict: Resultados de detección de cambios estructurales
    """
    results = {}
    
    if len(time_series) < 2 * min_size:
        results['change_points'] = []
        results['change_dates'] = []
        results['segments'] = 1
        return results
    
    # Método 1: Detección mediante PELT (Pruned Exact Linear Time)
    if RUPTURES_AVAILABLE:
        try:
            # Convertir a array numpy y manejo de valores nulos
            signal = time_series.fillna(method='ffill').fillna(method='bfill').values
            
            # Algoritmo PELT para detección óptima de puntos de cambio
            algo = rpt.Pelt(model="rbf").fit(signal)
            change_points = algo.predict(pen=10)
            
            # Filtrar puntos de cambio para asegurar segmentos mínimos
            if len(change_points) > 1:  # El último punto no es un cambio real
                change_points = change_points[:-1] 
            
            # Convertir índices a fechas si el índice es tipo fecha
            if isinstance(time_series.index, pd.DatetimeIndex):
                change_dates = [time_series.index[cp] for cp in change_points if cp < len(time_series)]
            else:
                change_dates = change_points
            
            results['change_points'] = change_points
            results['change_dates'] = change_dates
            results['segments'] = len(change_points) + 1
            
            # Calcular estadísticas pre y post punto de cambio para cada punto
            segment_stats = []
            for i, cp in enumerate(change_points):
                if cp >= len(time_series) or cp < min_size:
                    continue
                    
                # Definir segmentos anterior y posterior
                if i == 0:
                    pre_segment = time_series.iloc[:cp]
                else:
                    pre_segment = time_series.iloc[change_points[i-1]:cp]
                
                if i == len(change_points) - 1:
                    post_segment = time_series.iloc[cp:]
                else:
                    post_segment = time_series.iloc[cp:change_points[i+1]]
                
                # Calcular estadísticas para cada segmento
                pre_mean = pre_segment.mean()
                post_mean = post_segment.mean()
                
                # Realizar prueba estadística para comparar medias
                t_stat, p_value = stats.ttest_ind(
                    pre_segment.dropna(), 
                    post_segment.dropna(),
                    equal_var=False  # No asumimos varianzas iguales
                )
                
                segment_stats.append({
                    'change_point': cp,
                    'change_date': time_series.index[cp] if cp < len(time_series) else None,
                    'pre_mean': pre_mean,
                    'post_mean': post_mean,
                    'relative_change': (post_mean - pre_mean) / pre_mean if pre_mean != 0 else float('inf'),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            results['segment_stats'] = segment_stats
            
        except Exception as e:
            results['error'] = str(e)
            results['change_points'] = []
            results['change_dates'] = []
            results['segments'] = 1
    else:
        # Método alternativo si ruptures no está disponible: detección básica mediante rolling mean
        rolling_mean = time_series.rolling(window=min_size).mean()
        rolling_std = time_series.rolling(window=min_size).std()
        
        # Detectar cambios significativos en la media
        mean_diff = rolling_mean.diff().abs()
        threshold = rolling_std.mean() * 2  # 2 desviaciones estándar
        potential_cp = mean_diff[mean_diff > threshold].index.tolist()
        
        # Filtrar puntos de cambio cercanos (mínimo min_size pasos entre ellos)
        filtered_cp = []
        for cp in potential_cp:
            if not filtered_cp or (time_series.index.get_loc(cp) - 
                                  time_series.index.get_loc(filtered_cp[-1])) >= min_size:
                filtered_cp.append(cp)
        
        results['change_points'] = [time_series.index.get_loc(cp) for cp in filtered_cp]
        results['change_dates'] = filtered_cp
        results['segments'] = len(filtered_cp) + 1
    
    return results

def advanced_arima_modeling(time_series, exogenous=None, forecast_periods=12):
    """
    Realiza modelado avanzado ARIMA con selección automática de parámetros
    y pronóstico para series temporales epidemiológicas.
    
    Args:
        time_series (pd.Series): Serie temporal con casos epidemiológicos
        exogenous (pd.DataFrame, optional): Variables exógenas para ARIMAX
        forecast_periods (int): Número de períodos a pronosticar
        
    Returns:
        dict: Resultados del modelado incluyendo pronósticos e intervalos de confianza
    """
    results = {}
    
    # Asegurar estacionariedad si es necesario
    adf_result = adfuller(time_series.dropna())
    results['adf_test'] = {
        'statistic': adf_result[0],
        'pvalue': adf_result[1],
        'is_stationary': adf_result[1] < 0.05
    }

    # Determinar si es probable que exista estacionalidad
    acf_values = acf(time_series.dropna(), nlags=min(104, len(time_series)//2))
    
    # Buscar picos en la ACF que puedan indicar estacionalidad
    potential_seasonality = []
    for lag in [52, 26, 13, 12, 4]:  # Posibles periodicidades: anual, semestral, trimestral, mensual, semanal
        if lag < len(acf_values):
            if acf_values[lag] > 2 / np.sqrt(len(time_series)):  # Umbral de significancia
                potential_seasonality.append(lag)
    
    if not potential_seasonality:
        # Si no se detecta estacionalidad, usar el valor predeterminado anual (52 semanas)
        seasonal_period = 52
    else:
        # Usar el período con la correlación más alta
        seasonal_period = max(potential_seasonality, 
                              key=lambda lag: acf_values[lag] if lag < len(acf_values) else 0)
    
    results['detected_seasonality'] = {
        'period': seasonal_period,
        'potential_periods': potential_seasonality
    }
    
    # Ajustar modelo Auto-ARIMA si pmdarima está disponible
    if PMDARIMA_AVAILABLE and len(time_series.dropna()) >= max(2 * seasonal_period, 20):
        try:
            # Preparar datos
            train_data = time_series.dropna()
            
            # Ajustar modelo auto_arima con configuración avanzada
            model = auto_arima(
                train_data,
                exogenous=exogenous,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                m=seasonal_period,  # Período estacional detectado
                seasonal=True,
                d=None,  # Determinar automáticamente diferenciación
                D=None,  # Determinar automáticamente diferenciación estacional
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                information_criterion='aic'
            )
            
            # Guardar información del modelo
            results['model_info'] = {
                'order': model.order,
                'seasonal_order': model.seasonal_order,
                'aic': model.aic(),
                'bic': model.bic()
            }
            
            # Realizar pronóstico
            forecast, conf_int = model.predict(n_periods=forecast_periods, 
                                              return_conf_int=True,
                                              alpha=0.05)  # 95% de intervalo de confianza
            
            # Generar fechas futuras para el pronóstico
            if isinstance(time_series.index, pd.DatetimeIndex):
                last_date = time_series.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_periods,
                    freq=pd.infer_freq(time_series.index)
                )
            else:
                last_idx = time_series.index[-1]
                forecast_dates = range(last_idx + 1, last_idx + 1 + forecast_periods)
            
            # Construir DataFrame de pronóstico
            forecast_df = pd.DataFrame({
                'forecast': forecast,
                'lower_ci': conf_int[:, 0],
                'upper_ci': conf_int[:, 1]
            }, index=forecast_dates)
            
            results['forecast'] = forecast_df
            
            # Evaluar el ajuste del modelo a los datos históricos
            in_sample_pred = model.predict_in_sample()
            rmse = np.sqrt(np.mean((train_data.values - in_sample_pred) ** 2))
            mape = np.mean(np.abs((train_data.values - in_sample_pred) / train_data.values)) * 100
            
            results['model_fit'] = {
                'rmse': rmse,
                'mape': mape
            }
            
            # Identificar componentes de estacionalidad y tendencia
            if len(train_data) >= 2 * seasonal_period:
                stl_result = sm.tsa.seasonal_decompose(
                    train_data, 
                    period=seasonal_period,
                    extrapolate_trend='freq'
                )
                
                results['decomposition'] = {
                    'trend': stl_result.trend.tolist(),
                    'seasonal': stl_result.seasonal.tolist(),
                    'resid': stl_result.resid.tolist()
                }
            
        except Exception as e:
            results['error'] = str(e)
            # Implementar fallback al método SARIMA básico
            results.update(fit_basic_sarima(time_series, seasonal_period, forecast_periods))
    else:
        # Usar SARIMA básico si pmdarima no está disponible o la serie es demasiado corta
        results.update(fit_basic_sarima(time_series, seasonal_period, forecast_periods))
    
    return results

def fit_basic_sarima(time_series, seasonal_period=52, forecast_periods=12):
    """
    Ajusta un modelo SARIMA básico como fallback cuando auto_arima no está disponible.
    
    Args:
        time_series (pd.Series): Serie temporal de casos
        seasonal_period (int): Período de estacionalidad
        forecast_periods (int): Períodos a pronosticar
        
    Returns:
        dict: Resultados del modelo SARIMA
    """
    results = {}
    
    try:
        # Usar órdenes predeterminados razonables para epidemiología
        model = SARIMAX(
            time_series.dropna(),
            order=(1, 1, 1),  # p, d, q
            seasonal_order=(1, 1, 1, seasonal_period),  # P, D, Q, m
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False, maxiter=200)
        
        results['model_info'] = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, seasonal_period),
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
        
        # Realizar pronóstico
        forecast = fitted_model.get_forecast(steps=forecast_periods)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)  # 95% de intervalo de confianza
        
        # Generar fechas futuras para el pronóstico
        if isinstance(time_series.index, pd.DatetimeIndex):
            last_date = time_series.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq=pd.infer_freq(time_series.index)
            )
        else:
            last_idx = time_series.index[-1]
            forecast_dates = range(last_idx + 1, last_idx + 1 + forecast_periods)
        
        # Construir DataFrame de pronóstico
        forecast_df = pd.DataFrame({
            'forecast': mean_forecast.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        }, index=forecast_dates)
        
        results['forecast'] = forecast_df
        
        # Estadísticas de ajuste básicas
        results['model_fit'] = {
            'rmse': np.sqrt(fitted_model.mse),
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
        
    except Exception as e:
        results['error'] = str(e)
        results['forecast'] = None
    
    return results

def analyze_epidemic_cycles(df_weekly, outbreaks=None):
    """
    Realiza análisis de ciclos epidémicos basado en datos semanales
    e identificación previa de brotes.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        outbreaks (pd.DataFrame, optional): DataFrame con brotes identificados
        
    Returns:
        dict: Resultados del análisis de ciclos epidémicos
    """
    results = {}
    
    # Si no hay brotes identificados, intentar detectarlos
    if outbreaks is None or len(outbreaks) < 2:
        # Detectar picos como proxy de brotes
        cases_series = df_weekly['Casos'].fillna(0)
        threshold = cases_series.mean() + 1.5 * cases_series.std()
        
        # Encontrar picos usando scipy.signal
        peaks, _ = find_peaks(cases_series.values, height=threshold, distance=8)
        outbreak_dates = df_weekly.index[peaks] if len(peaks) > 0 else []
    else:
        outbreak_dates = outbreaks.index
    
    # Si hay menos de 2 brotes, no podemos analizar ciclos
    if len(outbreak_dates) < 2:
        results['cycle_detected'] = False
        results['n_outbreaks'] = len(outbreak_dates)
        return results
    
    # Calcular intervalos entre brotes
    intervals = []
    for i in range(1, len(outbreak_dates)):
        if isinstance(outbreak_dates[0], pd.Timestamp):
            # Calcular diferencia en días para fechas
            interval_days = (outbreak_dates[i] - outbreak_dates[i-1]).days
            intervals.append(interval_days)
        else:
            # Para índices numéricos, calcular diferencia directa
            intervals.append(outbreak_dates[i] - outbreak_dates[i-1])
    
    # Calcular estadísticas descriptivas de los intervalos
    intervals_array = np.array(intervals)
    
    results['cycle_detected'] = True
    results['n_outbreaks'] = len(outbreak_dates)
    results['intervals'] = {
        'days': intervals if isinstance(outbreak_dates[0], pd.Timestamp) else intervals,
        'mean': float(np.mean(intervals_array)),
        'median': float(np.median(intervals_array)),
        'std': float(np.std(intervals_array)),
        'min': float(np.min(intervals_array)),
        'max': float(np.max(intervals_array)),
        'cv': float(np.std(intervals_array) / np.mean(intervals_array))  # Coeficiente de variación
    }
    
    # Intentar identificar si hay un patrón cíclico claro
    if len(intervals) >= 3:
        # Calculamos el coeficiente de variación (CV) para medir la consistencia del ciclo
        cv = results['intervals']['cv']
        
        if cv < 0.2:  # CV bajo indica intervalos consistentes
            results['cycle_pattern'] = 'REGULAR'
            results['cycle_length'] = results['intervals']['mean']
            results['confidence'] = 'HIGH'
        elif cv < 0.4:
            results['cycle_pattern'] = 'MODERATELY_REGULAR'
            results['cycle_length'] = results['intervals']['mean']
            results['confidence'] = 'MEDIUM'
        else:
            results['cycle_pattern'] = 'IRREGULAR'
            results['cycle_length'] = results['intervals']['mean']
            results['confidence'] = 'LOW'
    else:
        results['cycle_pattern'] = 'INSUFFICIENT_DATA'
        results['cycle_length'] = None
        results['confidence'] = 'VERY_LOW'
    
    # Si hay un patrón cíclico, predecir el próximo brote
    if results['cycle_pattern'] in ['REGULAR', 'MODERATELY_REGULAR'] and len(outbreak_dates) > 0:
        last_outbreak = outbreak_dates[-1]
        cycle_length = results['cycle_length']
        
        if isinstance(last_outbreak, pd.Timestamp):
            # Para fechas, agregar días
            next_outbreak_estimate = last_outbreak + pd.Timedelta(days=cycle_length)
        else:
            # Para índices numéricos, sumar directamente
            next_outbreak_estimate = last_outbreak + cycle_length
        
        results['next_outbreak_estimate'] = next_outbreak_estimate
        
        # Calcular intervalo de predicción basado en la variabilidad de los ciclos
        if isinstance(last_outbreak, pd.Timestamp):
            prediction_interval = {
                'lower': last_outbreak + pd.Timedelta(days=cycle_length - 1.96 * results['intervals']['std']),
                'upper': last_outbreak + pd.Timedelta(days=cycle_length + 1.96 * results['intervals']['std'])
            }
        else:
            prediction_interval = {
                'lower': last_outbreak + cycle_length - 1.96 * results['intervals']['std'],
                'upper': last_outbreak + cycle_length + 1.96 * results['intervals']['std']
            }
        
        results['next_outbreak_interval'] = prediction_interval
    
    # Analizar estacionalidad de los brotes
    if isinstance(outbreak_dates[0], pd.Timestamp) and len(outbreak_dates) >= 4:
        # Extraer mes de cada brote
        outbreak_months = [d.month for d in outbreak_dates]
        month_counts = pd.Series(outbreak_months).value_counts()
        
        # Verificar si hay concentración en ciertos meses (estacionalidad)
        max_count = month_counts.max()
        total_outbreaks = len(outbreak_dates)
        max_proportion = max_count / total_outbreaks
        
        if max_proportion >= 0.4:  # 40% o más de los brotes en un solo mes
            results['seasonality'] = {
                'detected': True,
                'dominant_month': month_counts.idxmax(),
                'proportion': max_proportion
            }
            
            # Verificar si hay un período de 2-3 meses con alta concentración
            top_months = month_counts.nlargest(3)
            if (top_months.sum() / total_outbreaks) >= 0.7:  # 70% en 3 meses
                results['seasonality']['seasonal_period'] = top_months.index.tolist()
        else:
            results['seasonality'] = {
                'detected': False
            }
    
    return results

def compare_current_to_historical(time_series, current_period=4):
    """
    Compara las últimas semanas con el mismo período en años anteriores.
    
    Args:
        time_series (pd.Series): Serie temporal de casos
        current_period (int): Número de semanas recientes a comparar
        
    Returns:
        dict: Resultados de la comparación
    """
    if not isinstance(time_series.index, pd.DatetimeIndex):
        # No podemos hacer comparación histórica sin fechas
        return {'error': 'Se requiere índice de fechas para comparación histórica'}
    
    # Extraer las últimas N semanas
    current_data = time_series[-current_period:]
    current_mean = current_data.mean()
    
    # Obtener el mismo período en años anteriores
    historical_periods = []
    current_start = current_data.index[0]
    current_end = current_data.index[-1]
    
    max_years_back = 3  # Comparar hasta 3 años atrás
    comparison = {}
    
    for year_back in range(1, max_years_back + 1):
        historical_start = current_start - pd.DateOffset(years=year_back)
        historical_end = current_end - pd.DateOffset(years=year_back)
        
        historical_data = time_series[
            (time_series.index >= historical_start) & 
            (time_series.index <= historical_end)
        ]
        
        if len(historical_data) >= current_period / 2:  # Al menos la mitad de puntos
            historical_mean = historical_data.mean()
            percent_diff = ((current_mean - historical_mean) / historical_mean) * 100 if historical_mean > 0 else float('inf')
            
            comparison[f'{year_back}_year_ago'] = {
                'mean': float(historical_mean),
                'percent_difference': float(percent_diff)
            }
    
    # Determinar si la situación actual es anómala respecto al histórico
    if comparison:
        # Calcular promedio de todos los períodos históricos disponibles
        historical_means = [data['mean'] for data in comparison.values()]
        avg_historical = np.mean(historical_means)
        
        # Calcular desviación respecto al promedio histórico
        deviation = ((current_mean - avg_historical) / avg_historical) * 100 if avg_historical > 0 else float('inf')
        
        # Determinar estado basado en la desviación
        if deviation > 50:  # Más del 50% por encima del promedio histórico
            status = "ALERT"
        elif deviation > 20:  # Entre 20% y 50% por encima
            status = "WARNING"
        elif deviation < -20:  # Más del 20% por debajo
            status = "BELOW_AVERAGE"
        else:  # Entre -20% y 20%
            status = "NORMAL"
        
        comparison['current_mean'] = float(current_mean)
        comparison['avg_historical_mean'] = float(avg_historical)
        comparison['percent_deviation'] = float(deviation)
        comparison['status'] = status
    
    return comparison

def calculate_rt_proxy(time_series, window=7):
    """
    Calcula un proxy del número reproductivo efectivo (Rt) 
    basado en el cambio en casos a lo largo del tiempo.
    
    Args:
        time_series (pd.Series): Serie temporal de casos
        window (int): Tamaño de ventana para cálculo de Rt
        
    Returns:
        dict: Métricas de transmisibilidad
    """
    # Calcular cambio relativo en casos con un retraso (proxy simple de Rt)
    if len(time_series) <= window:
        return {'error': 'Serie temporal demasiado corta para cálculo de Rt'}
    
    # Suavizar la serie para reducir ruido
    smoothed = time_series.rolling(window=window, center=False).mean()
    
    # Calcular "pseudo-Rt" como la relación entre casos actuales y pasados
    rt_proxy = pd.Series(index=smoothed.index)
    
    for i in range(window, len(smoothed)):
        current = smoothed.iloc[i]
        previous = smoothed.iloc[i-window]
        
        if previous > 0:
            rt_value = (current / previous) ** (window/7)  # Normalizado a 1 semana
            rt_proxy.iloc[i] = rt_value
    
    # Calcular estadísticas de Rt para el período reciente
    recent_rt = rt_proxy.dropna()[-4:]  # Últimas 4 semanas
    
    return {
        'rt_proxy': rt_proxy.dropna().to_dict(),
        'current_rt': float(recent_rt.iloc[-1]) if len(recent_rt) > 0 else None,
        'recent_rt_mean': float(recent_rt.mean()) if len(recent_rt) > 0 else None,
        'recent_rt_std': float(recent_rt.std()) if len(recent_rt) > 0 else None,
        'above_threshold': (recent_rt > 1).mean() if len(recent_rt) > 0 else None
    }

def enhanced_time_series_analysis(df_weekly):
    """
    Implementa análisis de series temporales avanzado con detección de 
    cambios estructurales, modelado ARIMA y análisis de ciclos.
    
    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales
        
    Returns:
        dict: Resultados ampliados de análisis temporal
    """
    results = {}
    
    # Asegurar que tenemos datos suficientes
    if len(df_weekly) < 10:
        results['error'] = "Serie temporal demasiado corta para análisis avanzado"
        return results
    
    # Preparar serie de casos
    time_series = df_weekly['Casos'].copy()
    
    # 1. Detección de cambios estructurales en la serie
    structural_changes = detect_structural_changes(time_series)
    results['structural_changes'] = structural_changes
    
    # 2. Modelado avanzado ARIMA con pronóstico
    arima_results = advanced_arima_modeling(time_series)
    results.update(arima_results)
    
    # 3. Análisis de ciclos epidémicos
    outbreaks = results.get('outbreaks', None)
    cycle_analysis = analyze_epidemic_cycles(df_weekly, outbreaks)
    results['epidemic_cycles'] = cycle_analysis
    
    # 4. Indicadores de alerta temprana basados en dinamica actual
    current_period = 12  # Usar últimas 12 semanas como período actual
    
    if len(time_series) > current_period:
        current_data = time_series[-current_period:]
        previous_data = time_series[-(2*current_period):-current_period]
        
        # Calcular cambio porcentual en tendencia reciente
        current_trend = sm.OLS(current_data, sm.add_constant(np.arange(len(current_data)))).fit().params[1]
        
        # Solo calcular tendencia previa si hay suficientes datos
        if len(previous_data) >= current_period:
            previous_trend = sm.OLS(previous_data, sm.add_constant(np.arange(len(previous_data)))).fit().params[1]
            trend_change = ((current_trend - previous_trend) / previous_trend) * 100 if previous_trend != 0 else float('inf')
        else:
            previous_trend = None
            trend_change = None
        
        # Calcular velocidad de crecimiento
        current_mean = current_data.mean()
        previous_mean = previous_data.mean() if len(previous_data) > 0 else None
        
        if previous_mean is not None and previous_mean > 0:
            growth_rate = ((current_mean - previous_mean) / previous_mean) * 100
        else:
            growth_rate = None
        
        # Verificar si la tendencia reciente es significativa
        trend_model = sm.OLS(current_data, sm.add_constant(np.arange(len(current_data)))).fit()
        trend_significant = trend_model.pvalues[1] < 0.05
        
        results['early_warning'] = {
            'current_trend': current_trend,
            'previous_trend': previous_trend,
            'trend_change_percent': trend_change,
            'growth_rate': growth_rate,
            'trend_significant': trend_significant,
            'current_vs_historical': compare_current_to_historical(time_series)
        }
    
    # 5. Estadísticas de transmisibilidad (Rt proxies)
    if len(time_series) >= 14:  # Necesitamos al menos 2 semanas de datos
        # Calcular un proxy simple de Rt basado en la relación de casos consecutivos
        rt_proxy = calculate_rt_proxy(time_series)
        results['transmissibility_metrics'] = rt_proxy
    
    return results

def analyze_2024_in_detail(df, analysis_type):
    """
    Realiza un análisis detallado del año 2024.
    """
    df_2024 = df[df['Año'] == 2024].copy()
    analysis_2024 = {
        'total_cases': len(df_2024),
        'weekly_distribution': df_2024.groupby('Semana Epidemiologica').size().to_dict(),
        'age_distribution': df_2024['Edad_Anios'].describe().round(2).to_dict(),
        'gender_distribution': (df_2024['Sexo'].value_counts(normalize=True) * 100).round(2).to_dict(),
        'top_10_diagnoses': df_2024['Diagnostico Principal'].value_counts().head(10).to_dict(),
        'monthly_trend': df_2024.groupby('Mes').size().to_dict(),
        'severity_analysis': (df_2024['Destino'].value_counts(normalize=True) * 100).round(2).to_dict()
    }

    # Análisis específico según el tipo
    if analysis_type == 'respiratorio':
        analysis_2024['respiratory_specific'] = {
            'covid19_cases': len(df_2024[df_2024['CIE10 DP'].str.startswith('U07', na=False)]),
            'influenza_cases': len(df_2024[df_2024['CIE10 DP'].str.startswith(('J09', 'J10', 'J11'), na=False)]),
            'pneumonia_cases': len(df_2024[df_2024['CIE10 DP'].str.startswith(('J12', 'J13', 'J14', 'J15'), na=False)])
        }
    elif analysis_type == 'gastrointestinal':
        analysis_2024['gastrointestinal_specific'] = {
            'infectious_cases': len(df_2024[df_2024['CIE10 DP'].str.startswith(('A00', 'A01', 'A02', 'A03', 'A04'), na=False)]),
            'viral_cases': len(df_2024[df_2024['CIE10 DP'].str.startswith('A08', na=False)])
        }

    return analysis_2024

def analyze_2025_in_detail(df, analysis_type):
    """
    Realiza un análisis detallado del año 2025, considerando sus 53 semanas.

    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_type (str): Tipo de análisis ('respiratorio', 'gastrointestinal', 'varicela', 'manopieboca')

    Returns:
        dict: Diccionario con resultados del análisis detallado
    """
    df_2025 = df[df['Año'] == 2025].copy()

    # Análisis general
    analysis_2025 = {
        'total_cases': len(df_2025),
        'weekly_mean': round(df_2025.groupby('Semana Epidemiologica').size().mean(), 2),
        'weekly_median': round(df_2025.groupby('Semana Epidemiologica').size().median(), 2),
        'weekly_std': round(df_2025.groupby('Semana Epidemiologica').size().std(), 2)
    }

    # Distribución semanal
    weekly_distribution = df_2025.groupby('Semana Epidemiologica').agg({
        'CIE10 DP': 'count',
        'Edad_Anios': ['mean', 'std']
    }).round(2)

    analysis_2025['weekly_distribution'] = {
        week: {
            'cases': int(data['CIE10 DP']['count']),
            'mean_age': round(data['Edad_Anios']['mean'], 2),
            'std_age': round(data['Edad_Anios']['std'], 2) if pd.notna(data['Edad_Anios']['std']) else 0
        }
        for week, data in weekly_distribution.iterrows()
    }

    # Distribución por edad
    analysis_2025['age_distribution'] = df_2025['Edad_Anios'].describe().round(2).to_dict()

    # Distribución por género
    gender_dist = df_2025['Sexo'].value_counts(normalize=True)
    analysis_2025['gender_distribution'] = {gender: round(prop * 100, 2)
                                           for gender, prop in gender_dist.items()}

    # Top 10 diagnósticos
    analysis_2025['top_10_diagnoses'] = df_2025['Diagnostico Principal'].value_counts().head(10).to_dict()

    # Tendencia mensual
    analysis_2025['monthly_trend'] = df_2025.groupby('Mes').size().to_dict()

    # Análisis de severidad
    severity = df_2025['Destino'].value_counts(normalize=True)
    analysis_2025['severity_analysis'] = {dest: round(prop * 100, 2)
                                          for dest, prop in severity.items()}

    # Análisis específico de la semana 53
    df_week_53 = df_2025[df_2025['Semana Epidemiologica'] == 53]
    analysis_2025['week_53_analysis'] = {
        'total_cases': len(df_week_53),
        'age_distribution': df_week_53['Edad_Anios'].describe().round(2).to_dict(),
        'gender_distribution': df_week_53['Sexo'].value_counts(normalize=True).round(4).to_dict(),
        'severe_cases': len(df_week_53[df_week_53['Destino'] != 'DOMICILIO'])
    }

    # Análisis específico según tipo de enfermedad
    analysis_2025['disease_specific'] = {}

    if analysis_type == 'respiratorio':
        respiratory_analysis = {
            'influenza_cases': len(df_2025[df_2025['CIE10 DP'].str.startswith(('J09', 'J10', 'J11'), na=False)]),
            'pneumonia_cases': len(df_2025[df_2025['CIE10 DP'].str.startswith(('J12', 'J13', 'J14', 'J15'), na=False)]),
            'covid19_cases': len(df_2025[df_2025['CIE10 DP'].str.startswith('U07', na=False)]),
            'severity_rate': round(len(df_2025[df_2025['Destino'] != 'DOMICILIO']) / len(df_2025) * 100, 2) if len(df_2025) > 0 else 0
        }
        analysis_2025['disease_specific']['respiratory'] = respiratory_analysis

    elif analysis_type == 'gastrointestinal':
        gi_analysis = {
            'infectious_diarrhea_cases': len(df_2025[df_2025['CIE10 DP'].str.startswith(('A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'), na=False)]),
            'non_infectious_diarrhea_cases': len(df_2025[~df_2025['CIE10 DP'].str.startswith(('A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'), na=False)]),
            'severity_rate': round(len(df_2025[df_2025['Destino'] != 'DOMICILIO']) / len(df_2025) * 100, 2) if len(df_2025) > 0 else 0
        }
        analysis_2025['disease_specific']['gastrointestinal'] = gi_analysis

    elif analysis_type == 'varicela':
        varicela_analysis = {
            'under_5_cases': len(df_2025[df_2025['Edad_Anios'] < 5]),
            'complication_rate': round(len(df_2025[df_2025['Destino'] != 'DOMICILIO']) / len(df_2025) * 100, 2) if len(df_2025) > 0 else 0
        }
        analysis_2025['disease_specific']['varicela'] = varicela_analysis

    elif analysis_type == 'manopieboca':
        mpb_analysis = {
            'under_5_cases': len(df_2025[df_2025['Edad_Anios'] < 5]),
            'complication_rate': round(len(df_2025[df_2025['Destino'] != 'DOMICILIO']) / len(df_2025) * 100, 2) if len(df_2025) > 0 else 0
        }
        analysis_2025['disease_specific']['manopieboca'] = mpb_analysis

    # Comparación con años anteriores
    comparison_metrics = []
    for year in [2021, 2022, 2023, 2024]:
        df_year = df[df['Año'] == year]
        if len(df_year) > 0:
            year_cases = len(df_year)
            year_weekly_avg = year_cases / (53 if year == 2025 else 52)
            comparison_metrics.append({
                'year': year,
                'total_cases': year_cases,
                'weekly_average': round(year_weekly_avg, 2),
                'percent_difference': round((len(df_2025) - year_cases) / year_cases * 100, 2)
            })

    analysis_2025['year_comparison'] = comparison_metrics

    return analysis_2025

def analyze_time_series(df_weekly):
    """
    Analiza componentes de series temporales como estacionariedad, tendencia y componentes estacionales.
    
    Args:
        df_weekly (pandas.DataFrame): DataFrame con datos semanales indexados por fecha
        
    Returns:
        dict: Resultados del análisis de series temporales
    """
    results = {}
    
    try:
        # Test de estacionariedad (Augmented Dickey-Fuller)
        adf_test = adfuller(df_weekly['Casos'].dropna())
        
        results['adf_test'] = {
            'adf_statistic': adf_test[0],
            'p_value': adf_test[1],
            'used_lags': adf_test[2],
            'nobs': adf_test[3],
            'critical_values': adf_test[4],
            'is_stationary': adf_test[1] < 0.05
        }
        
        # Calcular componentes de tendencia, estacionalidad y residuos
        # Solo si hay suficientes puntos de datos para una descomposición significativa
        if len(df_weekly) >= 52:  # Al menos un año de datos
            try:
                # Intentar determinar una frecuencia estacional apropiada
                seasonal_periods = [52, 26, 13, 12]  # Potenciales períodos (anual, semestral, trimestral, mensual)
                
                # Usar autocorrelación para determinar mejor período
                acf_values = acf(df_weekly['Casos'].dropna(), nlags=max(seasonal_periods))
                
                # Buscar picos en ACF que sugieran estacionalidad
                best_period = 52  # Default a estacionalidad anual
                
                for period in seasonal_periods:
                    if period < len(acf_values) and acf_values[period] > 0.2:  # Umbral arbitrario para correlación
                        best_period = period
                        break
                
                # Descomponer la serie temporal con el período identificado
                decomposition = seasonal_decompose(
                    df_weekly['Casos'].dropna(),
                    period=best_period,
                    model='additive',
                    extrapolate_trend='freq'
                )
                
                results['decomposition'] = {
                    'period_used': best_period,
                    'trend': decomposition.trend.tolist() if hasattr(decomposition, 'trend') else [],
                    'seasonal': decomposition.seasonal.tolist() if hasattr(decomposition, 'seasonal') else [],
                    'resid': decomposition.resid.tolist() if hasattr(decomposition, 'resid') else [],
                }
                
                # Analizar la fuerza de la estacionalidad
                if hasattr(decomposition, 'seasonal') and hasattr(decomposition, 'resid'):
                    seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())
                    results['seasonality_strength'] = max(0, seasonal_strength)
                
                # Analizar la fuerza de la tendencia
                if hasattr(decomposition, 'trend') and hasattr(decomposition, 'resid'):
                    trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())
                    results['trend_strength'] = max(0, trend_strength)
            
            except Exception as e:
                results['decomposition_error'] = str(e)
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def perform_advanced_analysis(df, analysis_type):
    """
    Realiza un análisis estadístico avanzado de los datos epidemiológicos.
    """
    results = {}

    # Preparar datos semanales
    df_weekly = prepare_weekly_data(df)
    results['df_weekly'] = df_weekly

    # Análisis de tendencia
    results.update(analyze_trend(df_weekly))

    # Detección de outliers mejorada
    results['outliers'] = detect_outliers(df_weekly)

    # Análisis de estacionalidad con manejo dinámico del período
    results['seasonal_component'] = analyze_seasonality(df_weekly)

    # Modelado SARIMA adaptativo
    results['sarima_model'] = fit_sarima_model(df_weekly)

    # Detección de brotes mejorada
    results['outbreaks'] = detect_outbreaks(df_weekly)

    # Análisis de autocorrelación con período adaptativo
    results.update(analyze_autocorrelation(df_weekly))

    # Análisis por grupo etario
    results['weekly_cases_by_age_group'] = analyze_weekly_cases_by_age_group(df)

    # Análisis por diferentes grupos
    for group in ['Grupo_Edad', 'Estableciemiento', 'Comuna', 'Sexo', 'Pais_Origen']:
        results[f'{group.lower()}_analysis'] = analyze_group(df, group)

    # Análisis de diagnósticos
    results['top_10_diagnoses'] = analyze_diagnoses(df)

    # Análisis específico según tipo de enfermedad
    specific_analyses = {
        'respiratorio': analyze_respiratory_diseases,
        'gastrointestinal': analyze_gastrointestinal_diseases,
        'varicela': analyze_varicela_diseases,
        'manopieboca': analyze_manopieboca_diseases
    }

    if analysis_type in specific_analyses:
        disease_specific_results = specific_analyses[analysis_type](df)
        results[f'{analysis_type}_specific'] = disease_specific_results
        
        # Asegurarnos que el análisis por comuna esté disponible en el nivel superior
        if 'comuna_analysis' in disease_specific_results:
            results['comuna_analysis'] = disease_specific_results['comuna_analysis']

    # NUEVO: Análisis avanzado de series temporales
    time_series_results = enhanced_time_series_analysis(df_weekly)
    results['enhanced_time_series'] = time_series_results

    # Análisis adicionales
    results.update({
        'derivations': analyze_derivations(df),
        'annual_trends': analyze_annual_trends(df),
        'proportion_analysis': analyze_proportions(df),
        'correlation_analysis': analyze_correlations(df),
        'time_series_analysis': analyze_time_series(df_weekly),
        'detailed_2024_analysis': analyze_2024_in_detail(df, analysis_type),
        'detailed_2025_analysis': analyze_2025_in_detail(df, analysis_type),
        'establishment_analysis': analyze_by_establishment(df, analysis_type),
        'weekly_change': analyze_weekly_change(df)
    })

    return results

if __name__ == "__main__":
    print("Este módulo está diseñado para ser importado y utilizado en main.py")
    print("Para realizar el análisis estadístico, ejecute main.py")