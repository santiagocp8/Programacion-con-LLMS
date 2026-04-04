import pandas as pd
import numpy as np

def generar_caso_de_uso_imputar_nulos_por_grupo():
    n_rows = 15
    categorias = ["A", "B", "C"]
    df = pd.DataFrame({
        "sector": np.random.choice(categorias, n_rows),
        "consumo": np.random.uniform(100, 500, n_rows)
    })
    df.loc[np.random.choice(df.index, 3, replace=False), "consumo"] = np.nan
    
    # Resultado esperado
    df_out = df.copy()
    df_out["consumo"] = df_out.groupby("sector")["consumo"].transform(lambda x: x.fillna(x.median()))
    df_out["consumo"] = df_out["consumo"].fillna(df["consumo"].median())
    
    return {"df": df, "col_num": "consumo", "col_cat": "sector"}, df_out
