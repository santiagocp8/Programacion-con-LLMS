import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import HuberRegressor


def predecir_demanda_asistencia(df, target_col):
    # 1. Separar características numéricas de la variable objetivo
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
    y = df[target_col]

    # 2. Split 80/20 con random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Escalar con RobustScaler (robusto a outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Ajustar HuberRegressor (robusto a outliers)
    model = HuberRegressor()
    model.fit(X_train_scaled, y_train)

    # 5. Devolver R² en test
    return model.score(X_test_scaled, y_test)
