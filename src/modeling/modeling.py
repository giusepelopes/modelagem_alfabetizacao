import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing import preprocessing

# Desabilitar logs do Optuna para manter o terminal limpo durante os testes
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Obtendo o dataframe resultante a partir do preprocessamento.
data = preprocessing.preprocess()

drop_cols = ['ano', 'target_alfabetizado', 'feat_peso_aluno']

# Separação dos dados de treino e teste feita considerando ano de avaliação,
# Visando evitar vazamento de dados. Volumetria de dados próxima entre os dois anos.
data_2023 = data[data['ano'] == 2023].copy()
data_2024 = data[data['ano'] == 2024].copy()

X_dev = data_2023.drop(columns=drop_cols)
y_dev = data_2023['target_alfabetizado']

X_test = data_2024.drop(columns=drop_cols)
y_test = data_2024['target_alfabetizado']

# Split dos dados de treino para (80% treino / 20% validação interna).
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_dev
)

# Conversão das variáveis categóricas para uso na modelagem.
X_train['feat_uf'] = X_train['feat_uf'].astype('category')
X_val['feat_uf'] = X_val['feat_uf'].astype('category')
X_test['feat_uf'] = X_test['feat_uf'].astype('category')

X_train['feat_rede_encoded'] = X_train['feat_rede_encoded'].astype('category')
X_val['feat_rede_encoded'] = X_val['feat_rede_encoded'].astype('category')
X_test['feat_rede_encoded'] = X_test['feat_rede_encoded'].astype('category')

# Configuração dos datasets de treino e validação a serem usados para o LightGBM.
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

# Função Objetivo para o Optuna
def objective(trial):
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 130),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'feature_pre_filter': False
    }

    # Treinamento com Early Stopping
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Probabilidades na validação interna
    preds_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, preds_val_prob)
    return auc

# Execução da Otimização
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Treinamento guiado apenas pelos dados de validação interna (2023).
model = lgb.train(
    study.best_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

preds_validation = model.predict(X_val, num_iteration=model.best_iteration)

limiares = np.linspace(0.40, 0.75, 500)
melhor_limiar = 0.5
melhor_f05 = 0.0

for l in limiares:
    preds_binary = (preds_validation >= l).astype(int)
    # beta=0.42 penaliza mais os Falsos Positivos do que os Falsos Negativos
    score = fbeta_score(y_val, preds_binary, beta=0.42)

    if score > melhor_f05:
        melhor_f05 = score
        melhor_limiar = l

print('=== OTIMIZAÇÃO DE LIMIAR ===')
print(f'Limiar Padrão: 0.50')
print(f'Limiar Ótimo Encontrado: {melhor_limiar:.4f}')
print(f'Melhor F0.5-Score na Validação: {melhor_f05:.4f}\n')

preds_test_prob = model.predict(X_test, num_iteration=model.best_iteration)

# Comparativo Padrão vs. Ajustado
preds_test_default = (preds_test_prob >= 0.50).astype(int)
preds_test_adjusted = (preds_test_prob >= melhor_limiar).astype(int)

print('=== COMPARATIVO NO TESTE DE 2024 ===')
print(f'Precisão (Limiar 0.50): {precision_score(y_test, preds_test_default):.4f}')
print(f'Precisão (Limiar {melhor_limiar:.2f}): {precision_score(y_test, preds_test_adjusted):.4f}\n')

cm = confusion_matrix(y_test, preds_test_default)
vn, fp, fn, vp = cm.ravel()
print('Matriz de Confusão em 2024 (Limiar padrão):')
print(f'Verdadeiros Negativos (Não-Alfabetizados Corretos): {vn:,}')
print(f'Falsos Positivos (ERRO GRAVE: Previsto Alfabetizado, mas Não É): {fp:,}')
print(f'Falsos Negativos (Previsto Não-Alfabetizado, mas É): {fn:,}')
print(f'Verdadeiros Positivos (Alfabetizados Corretos): {vp:,}')

# Matriz de Confusão com Limiar Ajustado
cm = confusion_matrix(y_test, preds_test_adjusted)
vn, fp, fn, vp = cm.ravel()
print('Matriz de Confusão em 2024 (Limiar Ajustado):')
print(f'Verdadeiros Negativos (Não-Alfabetizados Corretos): {vn:,}')
print(f'Falsos Positivos (ERRO GRAVE: Previsto Alfabetizado, mas Não É): {fp:,}')
print(f'Falsos Negativos (Previsto Não-Alfabetizado, mas É): {fn:,}')
print(f'Verdadeiros Positivos (Alfabetizados Corretos): {vp:,}')

auc_2024 = roc_auc_score(y_test, preds_test_prob)
print(f'ROC-AUC Padrão em 2024: {auc_2024:.4f}')