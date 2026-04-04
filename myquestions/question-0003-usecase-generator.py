import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def generar_caso_de_uso_entrenar_clasificador_robusto():
    n_rows, n_feats = 20, 4
    X = pd.DataFrame(np.random.rand(n_rows, n_feats), columns=[f"f{i}" for i in range(n_feats)])
    y = np.random.randint(0, 2, n_rows)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=50))
    ])
    pipe.fit(X, y)
    
    return {"X": X, "y": y}, pipe
