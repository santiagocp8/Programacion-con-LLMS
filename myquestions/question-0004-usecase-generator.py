import numpy as np
from sklearn.metrics import f1_score

def generar_caso_de_uso_evaluar_f1_promedio():
    n_samples = 10
    clases = [0, 1, 2]
    y_true = np.random.choice(clases, n_samples)
    y_pred = np.random.choice(clases, n_samples)
    
    score = f1_score(y_true, y_pred, average='macro')
    
    return {"y_true": y_true, "y_pred": y_pred}, score
