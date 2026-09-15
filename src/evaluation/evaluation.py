import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.modeling import modeling
from src.visualization import visualization

best_model, best_threshold, X_test, y_test = modeling.train_model()

# Predição final utilizando os dados do conjunto de teste.
preds_test_prob = best_model.predict(X_test, num_iteration=best_model.best_iteration)
preds_test_final = (preds_test_prob >= best_threshold).astype(int)

#########################################
# Precisão e F0.5 com os dados de teste #
#########################################
print('[INFO:EVAL] AVALIAÇÃO NOS DADOS DE TESTE (ANO = 2024)')
print(f'\tLimiar Ótimo Aplicado: {best_threshold:.4f}')
print(f'\tPrecisão: {precision_score(y_test, preds_test_final):.4f}')
print(f'\tF0.5-Score: {fbeta_score(y_test, preds_test_final, beta=0.5):.4f}')

######################
# Matriz de confusão #
######################
cm = confusion_matrix(y_test, preds_test_final)
vn, fp, fn, vp = cm.ravel()
print('\n[INFO:EVAL] Matriz de Confusão:')
print(f'\tVerdadeiros Negativos: {vn:,}')
print(f'\tFalsos Positivos: {fp:,}')
print(f'\tFalsos Negativos: {fn:,}')
print(f'\tVerdadeiros Positivos: {vp:,}\n')

###########
# ROC_AUC #
###########
auc_2024 = roc_auc_score(y_test, preds_test_final)
print(f'[INFO:EVAL] ROC-AUC score: {auc_2024:.4f}')

print('\n[INFO:EVAL] Relatório de Classificação Completo:')
print(classification_report(y_test, preds_test_final, digits=4))

# Geração dos gráficos para visualização dos resultados.
visualization.results_visualization(best_model, best_threshold, preds_test_final, X_test, y_test)