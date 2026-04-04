import pandas as pd
import numpy as np

def generar_caso_de_uso_codificar_fechas_ciclicas():
    n_rows = np.random.randint(5, 10)
    fechas = pd.date_range(start="2026-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({"fecha_evento": fechas})
    
    # Resultado esperado
    df_out = df.copy()
    dias = df_out["fecha_evento"].dt.dayofweek
    df_out["dia_seno"] = np.sin(2 * np.pi * dias / 7)
    df_out["dia_coseno"] = np.cos(2 * np.pi * dias / 7)
    df_out = df_out.drop(columns=["fecha_evento"])
    
    return {"df": df, "col_fecha": "fecha_evento"}, df_out
