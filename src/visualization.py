import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import calendar
import os
from scipy import stats
from datetime import datetime

# Definir colores
ESTABLISHMENT_COLORS = {
    'SAPU LOMA COLORADA': 'red',
    'SAR  BOCA SUR': 'blue',
    'SAR SAN PEDRO': 'green'
}
ESTABLISHMENT_PALETTE = plt.cm.get_cmap('tab20')
YEAR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Colores para los años
AGE_GROUP_PALETTE = plt.cm.get_cmap('Set3')
HEATMAP_CMAP = 'YlGnBu'  # Paleta de colores para el heatmap
DEFAULT_PALETTE = sns.color_palette("deep")  # Paleta de colores por defecto

def get_year_max_weeks(year):
    """
    Obtiene el número máximo de semanas para un año específico.
    """
    return 53 if year == 2025 else 52

def save_plot(fig, filename, analysis_type):
    """
    Guarda el gráfico en la carpeta correspondiente.
    """
    folder = {
        'respiratorio': 'graphsrespiratorio',
        'gastrointestinal': 'graphsgastrointestinal',
        'varicela': 'graphsvaricela',
        'manopieboca': 'graphsmanopieboca'
    }.get(analysis_type, 'graphs')

    filepath = os.path.join('graphs', folder, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico guardado en: {filepath}")

def set_style():
    """
    Configura el estilo de los gráficos.
    """
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_weekly_cases_by_age_group(weekly_cases_by_age_group, analysis_type):
    """
    Grafica los casos semanales por grupo de edad, adaptado para diferentes longitudes de año.
    """
    age_groups = ['Menor de 1 A', '1 a 4 A', '5 a 14 A', '15 a 64 A', '65 y más A']

    for year, data in weekly_cases_by_age_group.items():
        max_weeks = get_year_max_weeks(year)

        fig, ax = plt.subplots(figsize=(20, 10))
        data_stacked = data.set_index('Semana')

        all_weeks = pd.DataFrame(index=range(1, max_weeks + 1))
        data_stacked = all_weeks.join(data_stacked).fillna(0)

        # Crear el gráfico de barras apiladas
        data_stacked[age_groups].plot(kind='bar', stacked=True, ax=ax, color=AGE_GROUP_PALETTE(np.linspace(0, 1, len(age_groups))))

        # Añadir etiquetas numéricas
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fontsize=8, color='white', weight='bold')

        ax.set_title(f'Atenciones Semanales por Rango Etario - {analysis_type.capitalize()} {year}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Semana Epidemiológica', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Atenciones', fontsize=12, fontweight='bold')

        ax.legend(title='Grupo Etario', bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=10, title_fontsize=12)

        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_plot(fig, f'weekly_cases_by_age_group_{analysis_type}_{year}.png', analysis_type)

def plot_establishment_comparison(df, analysis_type, establishment_analysis):
    """
    Grafica la comparación entre establecimientos, adaptado para diferentes longitudes de año.

    Args:
        df (pandas.DataFrame): DataFrame original
        analysis_type (str): Tipo de análisis
        establishment_analysis (dict): Resultados del análisis por establecimiento
    """
    set_style()

    # Usar un diccionario para recordar los colores asignados a cada establecimiento
    establishment_colors = {}
    color_palette = plt.cm.get_cmap('tab20')
    auto_color_index = 0

    for year, year_data in establishment_analysis.items():
        weekly_data = []
        max_weeks = get_year_max_weeks(year)

        for estab, estab_data in year_data.items():
            # Asignar color si no existe
            if estab not in establishment_colors:
                if estab in ESTABLISHMENT_COLORS:
                    establishment_colors[estab] = ESTABLISHMENT_COLORS[estab]
                else:
                    establishment_colors[estab] = color_palette(auto_color_index / 20)
                    auto_color_index += 1

            weekly_cases = estab_data.get('weekly_cases', {})
            for week in range(1, max_weeks + 1):
                weekly_data.append({
                    'Establecimiento': estab,
                    'Semana': week,
                    'Casos': weekly_cases.get(week, 0)  # Accede a la semana como entero
                })

        df_pivot = pd.DataFrame(weekly_data).pivot(
            index='Semana',
            columns='Establecimiento',
            values='Casos'
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(20, 10))
        num_establishments = len(df_pivot.columns)
        bar_width = 0.8 / num_establishments
        index = np.arange(len(df_pivot.index))

        for i, establishment in enumerate(df_pivot.columns):
            cases = df_pivot[establishment]
            position = index + i * bar_width
            # Usar el color asignado al establecimiento
            color = establishment_colors[establishment]

            bars = ax.bar(position, cases, bar_width,
                          label=establishment,
                          color=color)

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_title(f'Comparación de Casos Semanales {analysis_type.capitalize()} por Establecimiento - {year}',
                     fontweight='bold', fontsize=16)
        ax.set_xlabel('Semana Epidemiológica', fontweight='bold', fontsize=12)
        ax.set_ylabel('Número de Casos', fontweight='bold', fontsize=12)

        ax.set_xticks(index + bar_width * (num_establishments - 1) / 2)
        ax.set_xticklabels(range(1, max_weeks + 1), rotation=90, fontsize=8)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        ax.legend(title='Establecimientos', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        stats_text = []
        for estab, estab_data in year_data.items():
            stats_text.append(f"{estab}:")
            stats_text.append(f"  Total: {int(estab_data['total_cases'])}")
            stats_text.append(f"  Promedio edad: {estab_data['avg_age']:.1f}")
            stats_text.append(f"  % Casos graves: {estab_data['severity_rate']:.1f}%")
            stats_text.append("")

        plt.text(1.05, 0.5, '\n'.join(stats_text),
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 fontsize=10,
                 verticalalignment='center')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_plot(fig, f'establishment_comparison_{analysis_type}_{year}.png', analysis_type)

def plot_weekly_cases_by_year(df, analysis_type):
    """
    Grafica los casos semanales por año, adaptado para diferentes longitudes de año.
    """
    set_style()

    weekly_data = []
    for year in df['Año'].unique():
        df_year = df[df['Año'] == year]
        max_weeks = get_year_max_weeks(year)

        for week in range(1, max_weeks + 1):
            cases = df_year[df_year['Semana Epidemiologica'] == week]['CIE10 DP'].count()
            weekly_data.append({'Año': year, 'Semana': week, 'Casos': cases})

    df_weekly = pd.DataFrame(weekly_data)

    fig, ax = plt.subplots(figsize=(20, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_weekly['Año'].unique())))

    for i, year in enumerate(df_weekly['Año'].unique()):
        year_data = df_weekly[df_weekly['Año'] == year]

        # Filtrar puntos con valor mayor que 0
        year_data_filtered = year_data[year_data['Casos'] > 0]

        # Graficar la línea, solo donde Casos > 0
        line = ax.plot(year_data_filtered['Semana'], year_data_filtered['Casos'],
                       label=f'Año {year}',
                       color=YEAR_COLORS[i % len(YEAR_COLORS)], linewidth=2, marker='o', markersize=4)

        # Añadir etiquetas de valor a los puntos que son mayores que 0
        for x, y in zip(year_data['Semana'], year_data['Casos']):
            if y > 0:
                ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                            xytext=(0,10), ha='center', fontsize=8, rotation=45)

    ax.set_title(f'Casos Semanales de {analysis_type.capitalize()}\nSAR-SAPU San Pedro de la Paz - Período 2021-2025',
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Semana Epidemiológica', fontweight='bold', fontsize=12)
    ax.set_ylabel('Número de Casos', fontweight='bold', fontsize=12)

    # Ajustar el eje x para mostrar todas las semanas de cada año
    xticks = []
    xticklabels = []
    for year in sorted(df_weekly['Año'].unique()):
        max_weeks = get_year_max_weeks(year)
        xticks.extend(range(1, max_weeks + 1))
        xticklabels.extend([f"{week}" for week in range(1, max_weeks + 1)])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90, ha='right', fontsize=8)

    ax.legend(title='Año', title_fontsize='12', fontsize='10',
              loc='upper left', bbox_to_anchor=(1, 1))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Añadir líneas verticales para los trimestres y etiquetas Q
    for year in sorted(df_weekly['Año'].unique()):
        year_data = df_weekly[df_weekly['Año'] == year]
        max_week = year_data['Semana'].max()
        for quarter in [13, 26, 39]:
            if quarter <= max_week:
                ax.axvline(x=quarter, color='gray', linestyle='--', alpha=0.5)
                # Etiquetar los trimestres solo en el primer año
                if year == df_weekly['Año'].min():
                    ax.text(quarter, ax.get_ylim()[1], f'Q{(quarter-1)//13 + 1}',
                            ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Añadir estadísticas clave
    stats_text = (f"Estadísticas Clave:\n"
                  f"Promedio semanal: {df_weekly['Casos'].mean():.0f}\n"
                  f"Máximo semanal: {df_weekly['Casos'].max():.0f}\n"
                  f"Mínimo semanal: {df_weekly['Casos'].min():.0f}")

    ax.text(1.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_plot(fig, f'weekly_cases_by_year_{analysis_type}.png', analysis_type)

def plot_heatmap_cases_by_month_year(df, analysis_type):
    """
    Genera un heatmap de casos por mes y año.
    Muestra todos los valores, pero deja en blanco los meses que aún no han ocurrido.
    Utiliza la paleta de colores 'YlOrRd'.
    """
    set_style()

    monthly_cases = pd.pivot_table(
        df,
        values='CIE10 DP',
        index='Año',
        columns='Mes',
        aggfunc='count',
        fill_value=0
    )

    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    monthly_cases.columns = month_names

    fig, ax = plt.subplots(figsize=(15, 8))

    # Crear una matriz de anotaciones (valores a mostrar en cada celda)
    annot_matrix = monthly_cases.copy()
    current_year = datetime.now().year
    current_month = datetime.now().month

    for year in annot_matrix.index:
        for month_num in range(1, 13):
            # Si el año es mayor al actual o si el año es el actual pero el mes es mayor al actual,
            # entonces reemplaza el valor por una cadena vacía
            if year > current_year or (year == current_year and month_num > current_month):
                annot_matrix.loc[year, calendar.month_abbr[month_num]] = ''
            else:
                # Si no, formatear el valor existente como entero
                annot_matrix.loc[year, calendar.month_abbr[month_num]] = f"{int(monthly_cases.loc[year, calendar.month_abbr[month_num]]):,}"

    # Graficar el heatmap con la matriz de anotaciones personalizada y la paleta de colores 'YlOrRd'
    sns.heatmap(monthly_cases, annot=annot_matrix.values, fmt='', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Número de casos'}, linewidths=0, linecolor=None) # Se elimina formato a las celdas

    ax.set_title(f'Distribución Mensual de Casos - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz - Período 2021-2025',
                 fontweight='bold')
    ax.set_xlabel('Mes', fontweight='bold')
    ax.set_ylabel('Año', fontweight='bold')

    ax.set_xticklabels(month_names, rotation=0)

    plt.tight_layout()
    save_plot(fig, f'heatmap_cases_by_month_year_{analysis_type}.png', analysis_type)

def plot_cases_by_sex_and_age(df, analysis_type):
    set_style()

    df_plot = df.copy()
    df_plot['Edad_Anios'] = df_plot['Edad_Anios'].fillna(df_plot['Edad_Anios'].median())
    df_plot['Sexo'] = df_plot['Sexo'].fillna('NO ESPECIFICADO')

    fig, ax = plt.subplots(figsize=(15, 10))

    # Calcular pesos antes del bucle
    weights = np.ones_like(df_plot['Edad_Anios']) / len(df_plot)
    df_plot['Weights'] = weights

    # Usar sns.histplot pasando los datos en formato largo
    sns.histplot(
        data=df_plot,
        x='Edad_Anios',
        hue='Sexo',
        weights='Weights',  # Usar la columna de pesos
        bins=30,
        alpha=0.5,
        kde=True,
        ax=ax,
        stat="density",
        element="step",
        linewidth=0,
        palette="husl"
    )

    ax.set_title(f'Distribución de Casos por Sexo y Edad - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz - Período 2021-2025',
                 fontweight='bold')
    ax.set_xlabel('Edad (años)', fontweight='bold')
    ax.set_ylabel('Densidad', fontweight='bold')

    stats_text = []
    for sexo in df_plot['Sexo'].unique():
        data = df_plot[df_plot['Sexo'] == sexo]['Edad_Anios']
        stats_text.append(f'{sexo}:')
        stats_text.append(f'  Media: {data.mean():.1f} años')
        stats_text.append(f'  Mediana: {data.median():.1f} años')
        stats_text.append(f'  N: {len(data)}')

    plt.text(0.95, 0.95, '\n'.join(stats_text),
             transform=ax.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Sexo')

    plt.tight_layout()
    save_plot(fig, f'cases_by_sex_and_age_{analysis_type}.png', analysis_type)

def plot_trend_analysis(df_weekly, trend_results, analysis_type):
    """
    Genera un gráfico de análisis de tendencia con intervalos de confianza.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))

    # Filtrar los datos para excluir semanas con 0 casos o NaN en 'Casos'
    df_weekly_filtered = df_weekly[df_weekly['Casos'] > 0].dropna(subset=['Casos'])

    # Si no hay datos después de filtrar, mostrar un mensaje y retornar
    if df_weekly_filtered.empty:
        ax.text(0.5, 0.5, 'No hay datos para graficar (todos los valores son 0 o NaN)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Análisis de Tendencia - {analysis_type.capitalize()}\n'
                     f'SAR-SAPU San Pedro de la Paz',
                     fontweight='bold')
        ax.set_xlabel('Fecha', fontweight='bold')
        ax.set_ylabel('Número de Casos', fontweight='bold')
        plt.tight_layout()
        save_plot(fig, f'trend_analysis_{analysis_type}.png', analysis_type)
        return

    # Preparar datos para el gráfico
    dates = df_weekly_filtered.index
    casos = df_weekly_filtered['Casos'].values

    # Scatter plot de datos originales
    ax.scatter(dates, casos, alpha=0.5, label='Casos semanales', color='blue', s=30)

    # Generar línea de tendencia y su intervalo de confianza
    if isinstance(trend_results, dict) and 'trend' in trend_results:
        trend = trend_results['trend']
        mean_cases = np.mean(casos)

        # Calcular la línea de tendencia y el IC solo para las semanas filtradas
        X = np.arange(len(dates))
        trend_line = trend * X + mean_cases
        ax.plot(dates, trend_line, color='red', label='Línea de tendencia', linewidth=2)

        if 'trend_confidence_interval' in trend_results:
            ci_lower = trend_results['trend_confidence_interval'][0] * X + mean_cases
            ci_upper = trend_results['trend_confidence_interval'][1] * X + mean_cases
            ax.fill_between(dates, ci_lower, ci_upper, color='red', alpha=0.2, label='IC 95%')

        # Añadir texto con información de la tendencia
        trend_text = f"Tendencia: {'↑' if trend > 0 else '↓'} {abs(trend):.2f} casos/semana"
        if 'trend_pvalue' in trend_results:
            trend_text += f"\np-valor: {trend_results['trend_pvalue']:.4f}"

        ax.text(0.02, 0.98, trend_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)

    # Configuración del gráfico
    ax.set_title(f'Análisis de Tendencia - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Número de Casos', fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_plot(fig, f'trend_analysis_{analysis_type}.png', analysis_type)

def plot_outbreak_detection(df_weekly, outbreaks, analysis_type):
    """
    Genera un gráfico de detección de brotes adaptado para años con diferente número de semanas.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))

    # Filtrar los datos para excluir semanas con 0 casos
    df_weekly_filtered = df_weekly[df_weekly['Casos'] > 0]

    # Si no hay datos después de filtrar, mostrar un mensaje y retornar
    if df_weekly_filtered.empty:
        ax.text(0.5, 0.5, 'No hay datos para graficar (todos los valores son 0)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Detección de Brotes - {analysis_type.capitalize()}\n'
                     f'SAR-SAPU San Pedro de la Paz',
                     fontweight='bold')
        ax.set_xlabel('Fecha', fontweight='bold')
        ax.set_ylabel('Número de Casos', fontweight='bold')
        plt.tight_layout()
        save_plot(fig, f'outbreak_detection_{analysis_type}.png', analysis_type)
        return

    # Plotear serie temporal completa, solo con los puntos filtrados
    ax.plot(df_weekly_filtered.index, df_weekly_filtered['Casos'], label='Casos semanales', color='blue')

    # Marcar brotes detectados si existen y si las fechas de los brotes están en los datos filtrados
    if outbreaks is not None and len(outbreaks) > 0:
        outbreak_dates = []
        outbreak_cases = []

        if isinstance(outbreaks, pd.DataFrame):
            for date in outbreaks.index:
                if date in df_weekly_filtered.index:
                    outbreak_dates.append(date)
                    outbreak_cases.append(df_weekly_filtered.loc[date, 'Casos'])
        else:
            for date, cases in outbreaks.items():
                if date in df_weekly_filtered.index:
                    outbreak_dates.append(date)
                    outbreak_cases.append(cases)

        if outbreak_dates:
            ax.scatter(outbreak_dates, outbreak_cases,
                       color='red', s=100, label='Brotes detectados', zorder=5)

            # Añadir etiquetas para cada brote
            for date, cases in zip(outbreak_dates, outbreak_cases):
                ax.annotate(f'{int(cases)}',
                            xy=(date, cases),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->'),
                            fontsize=8)

    ax.set_title(f'Detección de Brotes - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Número de Casos', fontweight='bold')

    # Añadir información sobre umbral y método de detección
    ax.text(0.02, 0.98,
            'Método: Análisis de picos\nUmbral: Media + 2*DE',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_plot(fig, f'outbreak_detection_{analysis_type}.png', analysis_type)


def plot_autocorrelation(acf, pacf, analysis_type):
    """
    Genera gráficos de autocorrelación con manejo mejorado de lags.

    Args:
        acf (array-like): Valores de autocorrelación
        pacf (array-like): Valores de autocorrelación parcial
        analysis_type (str): Tipo de análisis ('respiratorio', 'gastrointestinal', etc.)
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Asegurar que tenemos arrays numpy
    acf_values = np.array(acf) if acf is not None else np.array([])
    pacf_values = np.array(pacf) if pacf is not None else np.array([])

    # Calcular lags y nivel de significancia
    lags = range(len(acf_values))
    significance_level = 1.96/np.sqrt(len(acf_values)) if len(acf_values) > 0 else 0.1

    # Plot ACF
    if len(acf_values) > 0:
        ax1.bar(lags, acf_values, width=0.2)
        ax1.axhline(y=0, linestyle='--', color='gray')
        ax1.axhline(y=significance_level, linestyle='--', color='red', alpha=0.5)
        ax1.axhline(y=-significance_level, linestyle='--', color='red', alpha=0.5)
        ax1.fill_between(lags, significance_level, -significance_level,
                         color='gray', alpha=0.2)

    ax1.set_title('Función de Autocorrelación (ACF)', fontweight='bold')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelación')

    # Plot PACF
    if len(pacf_values) > 0:
        ax2.bar(lags, pacf_values, width=0.2)
        ax2.axhline(y=0, linestyle='--', color='gray')
        ax2.axhline(y=significance_level, linestyle='--', color='red', alpha=0.5)
        ax2.axhline(y=-significance_level, linestyle='--', color='red', alpha=0.5)
        ax2.fill_between(lags, significance_level, -significance_level,
                         color='gray', alpha=0.2)

    ax2.set_title('Función de Autocorrelación Parcial (PACF)', fontweight='bold')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelación Parcial')

    # Añadir información sobre interpretación
    interpretation_text = (
        'Interpretación:\n'
        'Barras fuera de las líneas rojas\n'
        'indican correlaciones significativas\n'
        f'(nivel {significance_level:.3f})'
    )
    fig.text(0.98, 0.5, interpretation_text,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='center',
             horizontalalignment='right',
             fontsize=9)

    plt.tight_layout()
    save_plot(fig, f'autocorrelation_{analysis_type}.png', analysis_type)

def plot_proportion_analysis(proportion_results, analysis_type):
    """
    Genera gráficos de análisis de proporciones con intervalos de confianza.

    Args:
        proportion_results (dict): Resultados del análisis de proporciones
        analysis_type (str): Tipo de análisis ('respiratorio', 'gastrointestinal', etc.)
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # Función auxiliar para extraer proporciones e intervalos
    def extract_proportions(data_dict, category_prefix=''):
        categories = []
        props = []
        ci_lowers = []
        ci_uppers = []

        for category, stats in data_dict.items():
            if isinstance(stats, dict) and 'percentage' in stats:
                categories.append(f"{category_prefix}{category}")
                props.append(stats['percentage'])
                ci_lowers.append(stats['ci_lower'])
                ci_uppers.append(stats['ci_upper'])

        return categories, props, ci_lowers, ci_uppers

    # Graficar proporciones por género y grupo de edad
    categories1, props1, ci_lower1, ci_upper1 = extract_proportions(proportion_results.get('gender', {}), 'Género: ')
    categories2, props2, ci_lower2, ci_upper2 = extract_proportions(proportion_results.get('age_groups', {}), 'Edad: ')

    # Combinar datos
    all_categories = categories1 + categories2
    all_props = props1 + props2
    all_ci_lower = ci_lower1 + ci_lower2
    all_ci_upper = ci_upper1 + ci_upper2

    # Crear posiciones para las barras
    y_pos = np.arange(len(all_categories))

    # Graficar barras horizontales con errores en el primer subplot
    bars1 = ax1.barh(y_pos, all_props, align='center', color=plt.cm.viridis(np.linspace(0, 1, len(all_categories))))

    # Añadir barras de error
    ax1.errorbar(all_props, y_pos,
                 xerr=[np.array(all_props) - np.array(all_ci_lower),
                       np.array(all_ci_upper) - np.array(all_props)],
                 fmt='none', color='black', capsize=5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(all_categories)
    ax1.invert_yaxis()
    ax1.set_xlabel('Porcentaje (%)')
    ax1.set_title(f'Distribución por Género y Grupo Etario - {analysis_type.capitalize()}')

    # Añadir valores en las barras
    for i, v in enumerate(all_props):
        ax1.text(v + 1, i, f'{v:.1f}%', va='center')

    # Graficar proporciones de destino y establecimiento en el segundo subplot
    categories3, props3, ci_lower3, ci_upper3 = extract_proportions(proportion_results.get('destinations', {}), 'Destino: ')
    categories4, props4, ci_lower4, ci_upper4 = extract_proportions(proportion_results.get('establishments', {}), 'Estab.: ')

    all_categories2 = categories3 + categories4
    all_props2 = props3 + props4
    all_ci_lower2 = ci_lower3 + ci_lower4
    all_ci_upper2 = ci_upper3 + ci_upper4

    y_pos2 = np.arange(len(all_categories2))

    bars2 = ax2.barh(y_pos2, all_props2, align='center', color=plt.cm.viridis(np.linspace(0, 1, len(all_categories2))))

    ax2.errorbar(all_props2, y_pos2,
                 xerr=[np.array(all_props2) - np.array(all_ci_lower2),
                       np.array(all_ci_upper2) - np.array(all_props2)],
                 fmt='none', color='black', capsize=5)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(all_categories2)
    ax2.invert_yaxis()
    ax2.set_xlabel('Porcentaje (%)')
    ax2.set_title(f'Distribución por Destino y Establecimiento - {analysis_type.capitalize()}')

    # Añadir valores en las barras
    for i, v in enumerate(all_props2):
        ax2.text(v + 1, i, f'{v:.1f}%', va='center')

    # Ajustar diseño y agregar cuadrícula
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout()
    save_plot(fig, f'proportion_analysis_{analysis_type}.png', analysis_type)

def plot_weekly_change(weekly_change, analysis_type):
    """
    Genera gráfico de cambio semanal adaptado para diferentes longitudes de año.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))

    if not weekly_change:
        ax.text(0.5, 0.5, 'No hay datos disponibles para el análisis de cambio semanal',
                ha='center', va='center')
    else:
        for year in sorted(weekly_change.keys()):
            year_data = weekly_change[year]

            if isinstance(year_data, dict):
                changes = year_data.get('changes', {})
                if isinstance(changes, dict):
                    weeks = []
                    changes_values = []

                    if isinstance(changes, dict):
                        for week, change in changes.items():
                            if isinstance(week, int) and isinstance(change, (int, float)):
                                weeks.append(week)
                                changes_values.append(change)
                            elif isinstance(week, str) and week.isdigit() and isinstance(change, (int, float)):
                                weeks.append(int(week))
                                changes_values.append(change)

                    if weeks and changes_values:
                        # Ordenar por semana
                        weeks, changes_values = zip(*sorted(zip(weeks, changes_values)))

                        ax.plot(weeks, changes_values, marker='o',
                                label=f'Año {year}', linewidth=2)

                        # Añadir etiquetas para cambios significativos (ahora mayores a 30%)
                        for week, change in zip(weeks, changes_values):
                            if abs(change) > 30:
                                ax.annotate(f'{change:.1f}%',
                                            xy=(week, change),
                                            xytext=(0, 10),
                                            textcoords='offset points',
                                            ha='center',
                                            fontsize=8,
                                            arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_title(f'Cambio Porcentual Semanal de Casos - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Semana Epidemiológica', fontweight='bold')
    ax.set_ylabel('Cambio Porcentual (%)', fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    # Añadir línea de referencia en 0%
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_plot(fig, f'weekly_change_{analysis_type}.png', analysis_type)

# ----- NUEVAS FUNCIONES PARA VISUALIZACIONES DE SERIES TEMPORALES AVANZADAS -----

def plot_structural_changes(df_weekly, enhanced_ts_results, analysis_type):
    """
    Genera visualización de cambios estructurales en la serie temporal.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        analysis_type (str): Tipo de análisis epidemiológico
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))
    
    structural_changes = enhanced_ts_results.get('structural_changes', {})
    change_points = structural_changes.get('change_points', [])
    
    # Graficar serie temporal
    ax.plot(df_weekly.index, df_weekly['Casos'], label='Casos semanales', color='blue')
    
    # Marcar puntos de cambio
    if change_points:
        for cp in change_points:
            if cp < len(df_weekly):
                # Evitar índices fuera de rango
                date = df_weekly.index[cp]
                cases = df_weekly['Casos'].iloc[cp]
                
                # Marcar punto de cambio
                ax.axvline(x=date, color='red', linestyle='--', alpha=0.7)
                
                # Añadir etiqueta
                ax.annotate('Cambio', xy=(date, cases),
                           xytext=(10, 30),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10)
    
    # Agregar estadísticas de segmentos
    if 'segment_stats' in structural_changes:
        y_pos = ax.get_ylim()[1] * 0.9
        for i, stat in enumerate(structural_changes['segment_stats']):
            if 'change_point' in stat and stat['change_point'] < len(df_weekly):
                date = df_weekly.index[stat['change_point']]
                
                # Mostrar cambio relativo
                rel_change = stat.get('relative_change', 0)
                direction = '↑' if rel_change > 0 else '↓'
                
                ax.annotate(f"{direction} {abs(rel_change):.1%}",
                           xy=(date, y_pos - i * (y_pos * 0.05)),
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                           fontsize=10)
    
    ax.set_title(f'Análisis de Cambios Estructurales - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Número de Casos', fontweight='bold')
    
    # Leyenda y cuadrícula
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, f'structural_changes_{analysis_type}.png', analysis_type)

def plot_arima_forecast(df_weekly, enhanced_ts_results, analysis_type):
    """
    Genera visualización del pronóstico ARIMA.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        analysis_type (str): Tipo de análisis epidemiológico
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))
    
    forecast = enhanced_ts_results.get('forecast')
    
    if forecast is None or not isinstance(forecast, pd.DataFrame) or forecast.empty:
        ax.text(0.5, 0.5, 'Pronóstico no disponible',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        save_plot(fig, f'arima_forecast_{analysis_type}.png', analysis_type)
        return
    
    # Graficar datos históricos
    ax.plot(df_weekly.index, df_weekly['Casos'], label='Datos históricos', color='blue')
    
    # Graficar pronóstico
    ax.plot(forecast.index, forecast['forecast'], label='Pronóstico', color='red')
    
    # Graficar intervalo de confianza
    ax.fill_between(forecast.index, 
                   forecast['lower_ci'], 
                   forecast['upper_ci'],
                   color='red', alpha=0.2, label='IC 95%')
    
    # Añadir información del modelo
    model_info = enhanced_ts_results.get('model_info', {})
    if model_info:
        order = model_info.get('order', 'N/A')
        seasonal_order = model_info.get('seasonal_order', 'N/A')
        
        model_text = f"Modelo ARIMA{order}"
        if seasonal_order != 'N/A':
            model_text += f" x {seasonal_order}"
        
        # Añadir métricas de ajuste
        model_fit = enhanced_ts_results.get('model_fit', {})
        if model_fit:
            rmse = model_fit.get('rmse', 'N/A')
            if isinstance(rmse, (int, float)):
                model_text += f"\nRMSE: {rmse:.2f}"
                
            mape = model_fit.get('mape', 'N/A')
            if isinstance(mape, (int, float)):
                model_text += f", MAPE: {mape:.2f}%"
        
        ax.text(0.02, 0.95, model_text,
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    ax.set_title(f'Pronóstico ARIMA - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Número de Casos', fontweight='bold')
    
    # Añadir información sobre pronóstico
    if not forecast.empty:
        last_historical = df_weekly['Casos'].iloc[-1]
        first_forecast = forecast['forecast'].iloc[0]
        last_forecast = forecast['forecast'].iloc[-1]
        
        forecast_change = ((last_forecast - last_historical) / last_historical) * 100 if last_historical > 0 else float('inf')
        forecast_direction = "AUMENTO" if forecast_change > 0 else "DISMINUCIÓN"
        
        forecast_text = (f"Pronóstico a {len(forecast)} semanas:\n"
                        f"Valor inicial: {first_forecast:.1f}\n"
                        f"Valor final: {last_forecast:.1f}\n"
                        f"Cambio proyectado: {forecast_change:.1f}% ({forecast_direction})")
        
        ax.text(0.98, 0.05, forecast_text,
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='bottom',
               horizontalalignment='right',
               fontsize=10)
    
    # Leyenda y cuadrícula
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, f'arima_forecast_{analysis_type}.png', analysis_type)

def plot_epidemic_cycles(df_weekly, enhanced_ts_results, analysis_type):
    """
    Genera visualización del análisis de ciclos epidémicos.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        analysis_type (str): Tipo de análisis epidemiológico
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))
    
    epidemic_cycles = enhanced_ts_results.get('epidemic_cycles', {})
    
    # Si no hay ciclos detectados, mostrar mensaje y retornar
    if not epidemic_cycles.get('cycle_detected', False):
        ax.text(0.5, 0.5, 'No se detectaron ciclos epidémicos',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        save_plot(fig, f'epidemic_cycles_{analysis_type}.png', analysis_type)
        return
    
    # Graficar serie temporal
    ax.plot(df_weekly.index, df_weekly['Casos'], color='blue', alpha=0.7)
    
    # Marcar brotes detectados
    outbreaks = enhanced_ts_results.get('outbreaks')
    if isinstance(outbreaks, pd.DataFrame) and not outbreaks.empty:
        outbreak_dates = outbreaks.index
        outbreak_values = df_weekly.loc[outbreak_dates, 'Casos']
        
        ax.scatter(outbreak_dates, outbreak_values, color='red', s=100, 
                  label='Brotes detectados', zorder=5)
        
        # Dibujar líneas verticales en cada brote
        for date in outbreak_dates:
            ax.axvline(x=date, color='red', linestyle='--', alpha=0.3)
    
    # Agregar información sobre el ciclo
    intervals = epidemic_cycles.get('intervals', {})
    cycle_length = epidemic_cycles.get('cycle_length')
    cycle_pattern = epidemic_cycles.get('cycle_pattern')
    confidence = epidemic_cycles.get('confidence')
    
    # Crear texto con información del ciclo
    cycle_text = []
    
    if cycle_length is not None:
        cycle_text.append(f"Longitud del ciclo: {cycle_length:.1f} días")
    
    if 'mean' in intervals:
        cycle_text.append(f"Intervalo medio: {intervals['mean']:.1f} días")
    
    if 'std' in intervals:
        cycle_text.append(f"Desviación estándar: {intervals['std']:.1f} días")
    
    if cycle_pattern:
        pattern_desc = {
            'REGULAR': 'Regular',
            'MODERATELY_REGULAR': 'Moderadamente regular',
            'IRREGULAR': 'Irregular',
            'INSUFFICIENT_DATA': 'Datos insuficientes'
        }
        cycle_text.append(f"Patrón: {pattern_desc.get(cycle_pattern, cycle_pattern)}")
    
    if confidence:
        conf_desc = {
            'HIGH': 'Alta',
            'MEDIUM': 'Media',
            'LOW': 'Baja',
            'VERY_LOW': 'Muy baja'
        }
        cycle_text.append(f"Confianza: {conf_desc.get(confidence, confidence)}")
    
    # Agregar próximo brote estimado si está disponible
    if 'next_outbreak_estimate' in epidemic_cycles:
        next_estimate = epidemic_cycles['next_outbreak_estimate']
        
        if hasattr(next_estimate, 'strftime'):
            estimate_str = next_estimate.strftime('%Y-%m-%d')
        else:
            estimate_str = str(next_estimate)
        
        cycle_text.append(f"Próximo brote estimado: {estimate_str}")
        
        # Si hay datos sobre el intervalo de predicción
        if 'next_outbreak_interval' in epidemic_cycles:
            interval = epidemic_cycles['next_outbreak_interval']
            
            lower = interval['lower']
            upper = interval['upper']
            
            if hasattr(lower, 'strftime') and hasattr(upper, 'strftime'):
                lower_str = lower.strftime('%Y-%m-%d')
                upper_str = upper.strftime('%Y-%m-%d')
            else:
                lower_str = str(lower)
                upper_str = str(upper)
            
            cycle_text.append(f"Intervalo de predicción: {lower_str} - {upper_str}")
            
            # Marcar área del próximo brote estimado si está dentro del rango visible
            min_date = df_weekly.index.min()
            max_date = df_weekly.index.max()
            
            # Calcular rango de fechas visibles
            date_range = max_date - min_date
            extended_max = max_date + date_range * 0.2  # Extender 20% a la derecha
            
            # Verificar si la predicción está cerca del rango visible
            if lower <= extended_max:
                # Determinar el rango y de la predicción
                max_y = df_weekly['Casos'].max() * 1.1
                
                # Dibujar área de predicción
                if hasattr(lower, 'to_pydatetime'):
                    ax.axvspan(lower, upper, alpha=0.2, color='orange', 
                              label='Intervalo de predicción')
                    
                    # Agregar línea vertical para la estimación puntual
                    ax.axvline(x=next_estimate, color='orange', linestyle='-', 
                              label='Próximo brote estimado')
    
    # Mostrar información del ciclo en el gráfico
    if cycle_text:
        ax.text(0.02, 0.98, '\n'.join(cycle_text),
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    # Configurar título y etiquetas
    ax.set_title(f'Análisis de Ciclos Epidémicos - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Número de Casos', fontweight='bold')
    
    # Leyenda y cuadrícula
    ax.grid(True, linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, f'epidemic_cycles_{analysis_type}.png', analysis_type)

def plot_transmissibility_metrics(df_weekly, enhanced_ts_results, analysis_type):
    """
    Genera visualización de las métricas de transmisibilidad (proxy Rt).
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        analysis_type (str): Tipo de análisis epidemiológico
    """
    set_style()
    fig, ax = plt.subplots(figsize=(15, 8))
    
    transmissibility = enhanced_ts_results.get('transmissibility_metrics', {})
    
    if 'error' in transmissibility or 'rt_proxy' not in transmissibility:
        ax.text(0.5, 0.5, 'Métricas de transmisibilidad no disponibles',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        save_plot(fig, f'transmissibility_{analysis_type}.png', analysis_type)
        return
    
    # Convertir proxy de Rt a Serie de pandas
    rt_proxy = transmissibility['rt_proxy']
    
    if isinstance(rt_proxy, dict):
        dates = [pd.to_datetime(date) if isinstance(date, str) else date 
                for date in rt_proxy.keys()]
        values = list(rt_proxy.values())
        
        rt_series = pd.Series(values, index=dates)
    else:
        rt_series = pd.Series(rt_proxy)
    
    # Graficar Rt proxy
    ax.plot(rt_series.index, rt_series.values, color='purple', linewidth=2,
           label='Rt proxy (transmisibilidad)')
    
    # Añadir línea horizontal de referencia en Rt=1
    ax.axhline(y=1, color='red', linestyle='--', 
              label='Umbral epidémico (Rt=1)')
    
    # Colorear áreas según nivel de transmisibilidad
    ax.fill_between(rt_series.index, 1, rt_series.values, 
                   where=(rt_series.values > 1),
                   color='red', alpha=0.3, label='Crecimiento (Rt>1)')
    
    ax.fill_between(rt_series.index, rt_series.values, 1,
                   where=(rt_series.values < 1),
                   color='green', alpha=0.3, label='Decrecimiento (Rt<1)')
    
    # Agregar información sobre el Rt actual
    current_rt = transmissibility.get('current_rt')
    
    if current_rt is not None:
        rt_status = "CRECIMIENTO EPIDÉMICO" if current_rt > 1 else "DECRECIMIENTO EPIDÉMICO"
        rt_color = "red" if current_rt > 1 else "green"
        
        # Obtener la última fecha disponible
        last_date = rt_series.index[-1]
        
        # Marcador para el último Rt
        ax.scatter([last_date], [current_rt], color=rt_color, s=100, zorder=5)
        
        # Añadir etiqueta
        ax.annotate(f"Rt={current_rt:.2f}",
                   xy=(last_date, current_rt),
                   xytext=(10, 0),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                   fontsize=10)
        
        # Añadir texto con interpretación
        rt_text = [
            f"Rt actual: {current_rt:.2f}",
            f"Estado: {rt_status}",
        ]
        
        # Agregar Rt promedio reciente si está disponible
        rt_mean = transmissibility.get('recent_rt_mean')
        if rt_mean is not None:
            rt_text.append(f"Rt promedio reciente: {rt_mean:.2f}")
        
        # Agregar proporción de tiempo con Rt>1 si está disponible
        above_threshold = transmissibility.get('above_threshold')
        if above_threshold is not None:
            rt_text.append(f"Proporción de tiempo con Rt>1: {above_threshold:.0%}")
        
        # Mostrar cuadro de texto con información
        ax.text(0.02, 0.98, '\n'.join(rt_text),
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    # Configurar título y etiquetas
    ax.set_title(f'Número Reproductivo Efectivo (Rt) - {analysis_type.capitalize()}\n'
                 f'SAR-SAPU San Pedro de la Paz',
                 fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Rt (transmisibilidad)', fontweight='bold')
    
    # Configurar eje y para mostrar valores relevantes
    ax.set_ylim(0, max(3, rt_series.max() * 1.1))
    
    # Leyenda y cuadrícula
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, f'transmissibility_{analysis_type}.png', analysis_type)

def plot_enhanced_time_series(df_weekly, enhanced_ts_results, analysis_type):
    """
    Genera visualizaciones para el análisis avanzado de series temporales.
    
    Args:
        df_weekly (pd.DataFrame): DataFrame con datos semanales
        enhanced_ts_results (dict): Resultados del análisis avanzado de series temporales
        analysis_type (str): Tipo de análisis epidemiológico
    """
    if not enhanced_ts_results:
        return
    
    # 1. Visualización de cambios estructurales
    plot_structural_changes(df_weekly, enhanced_ts_results, analysis_type)
    
    # 2. Visualización de pronóstico ARIMA
    plot_arima_forecast(df_weekly, enhanced_ts_results, analysis_type)
    
    # 3. Visualización de ciclos epidémicos
    plot_epidemic_cycles(df_weekly, enhanced_ts_results, analysis_type)
    
    # 4. Visualización de métricas de transmisibilidad
    plot_transmissibility_metrics(df_weekly, enhanced_ts_results, analysis_type)

# Estas funciones específicas deben ser implementadas según los gráficos que se deseen generar para cada tipo de análisis
def plot_respiratory_specific(df, analysis_results):
    """
    Genera visualizaciones específicas para el análisis respiratorio.
    """
    pass  # Implementar la lógica para generar gráficos específicos

def plot_gastrointestinal_specific(df, analysis_results):
    """
    Genera visualizaciones específicas para el análisis gastrointestinal.
    """
    pass  # Implementar la lógica para generar gráficos específicos

def plot_varicela_specific(df, analysis_results):
    """
    Genera visualizaciones específicas para el análisis de varicela.
    """
    pass  # Implementar la lógica para generar gráficos específicos

def plot_manopieboca_specific(df, analysis_results):
    """
    Genera visualizaciones específicas para el análisis de mano-pie-boca.
    """
    pass  # Implementar la lógica para generar gráficos específicos

def create_visualizations(df, analysis_type, analysis_results):
    """
    Función principal que genera todas las visualizaciones.

    Args:
        df (pandas.DataFrame): DataFrame con los datos
        analysis_type (str): Tipo de análisis ('respiratorio', 'gastrointestinal', etc.)
        analysis_results (dict): Resultados del análisis estadístico
    """
    try:
        print(f"Generando visualizaciones para análisis {analysis_type}...")

        # Crear directorio si no existe
        os.makedirs(os.path.join('graphs', f'graphs{analysis_type}'), exist_ok=True)

        # 1. Visualizaciones generales
        plot_heatmap_cases_by_month_year(df, analysis_type)
        plot_cases_by_sex_and_age(df, analysis_type)
        plot_weekly_cases_by_year(df, analysis_type)

        # 2. Visualizaciones de análisis temporal
        if 'df_weekly' in analysis_results:
            plot_trend_analysis(analysis_results['df_weekly'], analysis_results, analysis_type)
            plot_outbreak_detection(analysis_results['df_weekly'], analysis_results['outbreaks'], analysis_type)
            plot_autocorrelation(analysis_results.get('acf',[]), analysis_results.get('pacf',[]), analysis_type)

            # NUEVO: Visualizaciones avanzadas de series temporales
            if 'enhanced_time_series' in analysis_results:
                plot_enhanced_time_series(analysis_results['df_weekly'], 
                                         analysis_results['enhanced_time_series'], 
                                         analysis_type)

        # 3. Visualizaciones de análisis estructural
        if 'proportion_analysis' in analysis_results:
            plot_proportion_analysis(analysis_results['proportion_analysis'], analysis_type)

        if 'establishment_analysis' in analysis_results:
            plot_establishment_comparison(df, analysis_type, analysis_results['establishment_analysis'])

        if 'weekly_cases_by_age_group' in analysis_results:
            plot_weekly_cases_by_age_group(analysis_results['weekly_cases_by_age_group'], analysis_type)

        if 'weekly_change' in analysis_results:
            plot_weekly_change(analysis_results['weekly_change'], analysis_type)

        # 4. Visualizaciones específicas por tipo de enfermedad
        specific_plot_functions = {
            'respiratorio': plot_respiratory_specific,
            'gastrointestinal': plot_gastrointestinal_specific,
            'varicela': plot_varicela_specific,
            'manopieboca': plot_manopieboca_specific,
        }

        if analysis_type in specific_plot_functions:
            specific_plot_functions[analysis_type](df, analysis_results)

        print(f"Visualizaciones para {analysis_type} generadas exitosamente.")

    except Exception as e:
        print(f"Error al generar visualizaciones: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Este módulo está diseñado para ser importado y utilizado en main.py")
    print("Para realizar el análisis y generar visualizaciones, ejecute main.py")