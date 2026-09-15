import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score)
import shap

# SHAP values calculados para uma amostra de 100000 para reduzir o tempo de cálculo,
# Visto que temos muitos registros no conjunto de teste.
def results_visualization(best_model, best_threshold, preds_test_final, X_test, y_test):
    
    X_test_sample = X_test.sample(100000, random_state=42)
    
    # Criação do SHAP explainer
    explainer = shap.TreeExplainer(best_model)

    # Cálculo dos valores SHAP para o conjunto de teste.
    # Utilizando uma amostra para reduzir o tempo de geração dos gráficos.
    shap_values = explainer(X_test_sample)

    # Gráfico SHAP values Beeswarm.
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, max_display=15)
    
    # Gráfico SHAP values de barras.
    plt.figure(figsize=(10, 6), layout="constrained")
    shap.plots.bar(shap_values, max_display=15)
    
    # Gráfico com os dados da matriz de confusão resultante.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=preds_test_final, display_labels=['Não-Alfabetizado (0)', 'Alfabetizado (1)'], cmap='Blues', values_format=',d', ax=axes[0])

    axes[0].set_title(f'Matriz de Confusão\nLimiar de Decisão: {best_threshold:.2f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predição do Modelo', fontsize=10)
    axes[0].set_ylabel('Classe Real', fontsize=10)
    axes[0].grid(False)
    
    # Gráfico com a ROC-AUC para as predições obtidas.
    auc_value = roc_auc_score(y_test, preds_test_final)
    RocCurveDisplay.from_predictions(y_true=y_test,y_score=preds_test_final,name='LightGBM',ax=axes[1])

    axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', label='Aleatório (AUC = 0.50)')
    axes[1].set_title(f'Curva ROC\nROC-AUC = {auc_value:.4f}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Taxa de Falsos Positivos', fontsize=10)
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos', fontsize=10)
    axes[1].legend(loc='lower right')

    # Exibir os gráficos lado a lado
    plt.tight_layout()
    plt.show()