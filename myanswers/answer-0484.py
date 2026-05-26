import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def seleccionar_features_por_peso(X, y, porcentaje_acumulado):
    # 1. Entrenar RandomForest para obtener importancias
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # 2. Ordenar importancias de mayor a menor
    importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # 3. Suma acumulada y punto de corte
    acumulada = np.cumsum(importancias.values)
    indice_corte = np.where(acumulada >= porcentaje_acumulado)[0][0]

    # 4. Devolver array con los nombres seleccionados
    return importancias.index[:indice_corte + 1].values
