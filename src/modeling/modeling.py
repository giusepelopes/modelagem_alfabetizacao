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

# Responsável por executar o preprocessamento e retornar os dados do split train -> validation -> test
def get_modeling_data():
    # Obtendo o dataframe resultante a partir do preprocessamento.
    data = preprocessing.preprocess()
    
    print("[INFO:MODEL] Executando a divisão de dados train -> validation -> test.")
    
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
    for col in ['feat_uf', 'feat_rede_encoded']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    return X_train, X_val, X_test, y_train, y_val, y_test 

# Função objetivo para o Optuna, implementada visando otimização para F0.5 Score,
# visto que vamos priorizar precisão como métrica, para minimização de falsos positivos.
def objective(trial, dtrain, dval, X_val, y_val):
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 80),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'feature_pre_filter': False
    }

    # Treinamento com Early Stopping
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Predição com os valores do conjunto de validação
    preds_val_prob = model.predict(X_val, num_iteration=model.best_iteration)

    # Busca do melhor limiar focado em F0.5-Score
    thresholds = np.linspace(0.50, 0.80, 30)
    best_f05 = 0.0
    for t in thresholds:
        preds_binary = (preds_val_prob >= t).astype(int)
        score = fbeta_score(y_val, preds_binary, beta=0.5, zero_division=0)
        if score > best_f05:
            best_f05 = score

    return best_f05

# Função responsável por executar o método estudo do Optuna, para otimizar hiperparâmetros.
def optimize_hp(dtrain, dval, X_val, y_val):
    print(f'[INFO:MODEL] Iniciando otimização de hiperparâmetros.')

    # Execução da otimização com 30 tentativas
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dtrain, dval, X_val, y_val), n_trials=30, show_progress_bar=True)

    print(f'[INFO:MODEL] Otimização de hyperparâmetros completa.')
    
    return study

def train_model():
    
    X_train, X_val, X_test, y_train, y_val, y_test = get_modeling_data()

    # Configuração dos datasets de treino e validação a serem usados para o LightGBM.
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    study = optimize_hp(dtrain, dval, X_val, y_val)

    print(f'[INFO:MODEL] Prosseguindo com treinamento do modelo utilizando hiperparâmetros ótimos encontrados.')
    # Re-treinamento utilizando os parametros ótimos encontrados pelo optuna.
    best_params = study.best_params
    best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1})

    best_model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Encontrar o limiar definitivo na validação de 2023
    preds_val_prob = best_model.predict(X_val, num_iteration=best_model.best_iteration)

    best_threshold = 0.5
    max_f05 = 0.0
    for threshold in np.linspace(0.50, 0.80, 50):
        score = fbeta_score(y_val, (preds_val_prob >= threshold).astype(int), beta=0.5, zero_division=0)
        if score > max_f05:
            max_f05 = score
            best_threshold = threshold

    print(f'[INFO:MODEL] Treinamento concluído.\n')
    return best_model, best_threshold, X_test, y_test
