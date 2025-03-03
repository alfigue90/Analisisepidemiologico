import pandas as pd
import numpy as np
from datetime import datetime
import re

# Definición actualizada de los datos de los años a procesar
datos_anos = {
    "2021": {
        "archivo": "2021.txt",
        "inicio_anio": datetime(2021, 1, 3),
        "fin_anio": datetime(2022, 1, 1),
        "max_semanas": 52
    },
    "2022": {
        "archivo": "2022.txt",
        "inicio_anio": datetime(2022, 1, 2),
        "fin_anio": datetime(2022, 12, 31),
        "max_semanas": 52
    },
    "2023": {
        "archivo": "2023.txt",
        "inicio_anio": datetime(2023, 1, 1),
        "fin_anio": datetime(2023, 12, 30),
        "max_semanas": 52
    },
    "2024": {
        "archivo": "2024.txt",
        "inicio_anio": datetime(2023, 12, 31),
        "fin_anio": datetime(2024, 12, 28),
        "max_semanas": 52
    },
    "2025": {
        "archivo": "2025.txt",
        "inicio_anio": datetime(2024, 12, 29),
        "fin_anio": datetime(2026, 1, 3),
        "max_semanas": 53
    }
}

def get_day_of_week(fecha):
    """
    Obtiene el día de la semana de manera segura, funcionando tanto con datetime como con pandas Timestamp.
    0 = Lunes, 6 = Domingo
    """
    if hasattr(fecha, 'weekday'):
        return fecha.weekday()
    elif isinstance(fecha, pd.Timestamp):
        return fecha.dayofweek
    else:
        return pd.Timestamp(fecha).dayofweek

def calcular_semana_epidemiologica(fecha, inicio_anio, fin_anio, max_semanas):
    """
    Calcula la semana epidemiológica para una fecha dada.
    """
    if fecha < inicio_anio or fecha >= fin_anio:
        return None
    
    dias_transcurridos = (fecha - inicio_anio).days
    semana = dias_transcurridos // 7 + 1
    
    return min(semana, max_semanas)

def load_and_preprocess_data(file_path, year, analysis_type):
    """
    Carga y preprocesa los datos de un archivo para un año específico.
    """
    try:
        # Obtener configuración del año
        year_config = datos_anos[str(year)]
        inicio_anio = year_config["inicio_anio"]
        fin_anio = year_config["fin_anio"]
        max_semanas = year_config["max_semanas"]
        
        # Cargar datos con manejo explícito de tipos
        df = pd.read_csv(file_path, sep='\t', 
                        dtype={
                            'Fecha Admision': str,
                            'Estableciemiento': str,
                            'Comuna': str,
                            'Edad': 'Int64',
                            'Tipo': str,
                            'Sexo': str,
                            'CIE10 DP': str,
                            'Diagnostico Principal': str,
                            'Destino': str,
                            'Pais_Origen': str
                        },
                        low_memory=False)
        
        # Convertir fechas
        df['Fecha Admision'] = pd.to_datetime(df['Fecha Admision'], format='%d-%m-%Y')
        
        # Manejar valores nulos
        df['CIE10 DP'] = df['CIE10 DP'].fillna('')
        df['Pais_Origen'] = df['Pais_Origen'].fillna('DESCONOCIDO')
        df['Edad'] = df['Edad'].fillna(-1)

        # Filtrar por tipo de análisis
        if analysis_type == 'respiratorio':
            respiratory_pattern = r'^(J0[0-9X]{2}|J1[0-9X]{2}|J4[0-6][0-9X]|J470|U071|U072)$'
            df = df[df['CIE10 DP'].str.match(respiratory_pattern)]
        elif analysis_type == 'gastrointestinal':
            df = df[df['CIE10 DP'].apply(lambda x: x.startswith('A') and x[1:4].isdigit() and 0 <= int(x[1:4]) <= 99)]
        elif analysis_type == 'varicela':
            df = df[df['CIE10 DP'] == 'B019']
        elif analysis_type == 'manopieboca':
            df = df[df['CIE10 DP'] == 'B084']
        
        # Calcular semana epidemiológica
        df['Semana Epidemiologica'] = df['Fecha Admision'].apply(
            lambda x: calcular_semana_epidemiologica(x, inicio_anio, fin_anio, max_semanas)
        )
        
        # Filtrar fechas fuera de rango
        df = df.dropna(subset=['Semana Epidemiologica'])
        
        # Añadir columnas temporales
        df['Año'] = year
        df['Mes'] = df['Fecha Admision'].dt.month
        df['Dia_semana'] = df['Fecha Admision'].apply(get_day_of_week)
        
        # Calcular edad en años
        df['Edad_Anios'] = np.where(df['Tipo'] == 'A', df['Edad'],
                                   np.where(df['Tipo'] == 'M', df['Edad']/12,
                                          df['Edad']/365))
        
        # Asignar grupos de edad
        def asignar_grupo_edad(edad, tipo):
            if edad == -1:
                return 'Desconocido'
            if tipo in ['M', 'D'] or (tipo == 'A' and edad <= 0):
                return 'Menor de 1 A'
            elif 1 <= edad <= 4:
                return '1 a 4 A'
            elif 5 <= edad <= 14:
                return '5 a 14 A'
            elif 15 <= edad <= 64:
                return '15 a 64 A'
            else:
                return '65 y más A'

        df['Grupo_Edad'] = df.apply(lambda row: asignar_grupo_edad(row['Edad'], row['Tipo']), axis=1)
        
        # Limpiar campos de texto
        text_columns = ['Estableciemiento', 'Comuna', 'Sexo', 'Diagnostico Principal', 'Destino', 'Pais_Origen']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].str.upper().str.strip()
        
        # Verificar y completar semanas faltantes
        semanas_unicas = df['Semana Epidemiologica'].nunique()
        if semanas_unicas != max_semanas:
            semanas_existentes = set(df['Semana Epidemiologica'].unique())
            semanas_esperadas = set(range(1, max_semanas + 1))
            semanas_faltantes = semanas_esperadas - semanas_existentes
            
            if semanas_faltantes:
                registros_faltantes = []
                for semana in semanas_faltantes:
                    fecha_semana = pd.Timestamp(inicio_anio) + pd.Timedelta(weeks=semana-1)
                    registros_faltantes.append({
                        'Fecha Admision': fecha_semana,
                        'Semana Epidemiologica': semana,
                        'Año': year,
                        'Mes': fecha_semana.month,
                        'Dia_semana': get_day_of_week(fecha_semana)
                    })
                if registros_faltantes:
                    df_temp = pd.DataFrame(registros_faltantes)
                    df = pd.concat([df, df_temp], ignore_index=True)
        
        # Ordenar el DataFrame
        df = df.sort_values(['Año', 'Semana Epidemiologica'])
        
        return df

    except Exception as e:
        print(f"Error detallado al procesar el archivo {file_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
   
def detect_missing_data(df):
    """
    Detecta datos faltantes en el DataFrame.
    """
    return (df.isnull().sum() / len(df)) * 100

if __name__ == "__main__":
    print("Este módulo está diseñado para ser importado y utilizado en main.py")
    print("Para realizar el procesamiento de datos, ejecute main.py")