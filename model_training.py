import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_processed_data(file_path: str = 'processed_data.parquet'):
    """Carrega os dados processados pelo Módulo 1."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado. Execute o Módulo 1 primeiro.")
    
    df = pd.read_parquet(file_path)
    meta_cols = ['company_name', 'year', 'target']
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, feature_cols

def split_and_balance_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.30, random_state: int = 42):
    """
    Executa a divisão estratificada (Hold-out) e aplica SMOTE no conjunto de treino.
    """
    print(f"-> Executando divisão Hold-out ({int((1-test_size)*100)}% Treino / {int(test_size*100)}% Teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Treino Original: {X_train.shape[0]} amostras | Falências: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"   Teste Original:  {X_test.shape[0]} amostras | Falências: {y_test.sum()} ({y_test.mean():.2%})")
    
    # Aplicação de SMOTE apenas nos dados de treino
    print("-> Aplicando SMOTE no conjunto de treinamento...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   Treino Pós-SMOTE: {X_train_res.shape[0]} amostras | Falências: {y_train_res.sum()} ({y_train_res.mean():.2%})")
    
    return X_train, X_train_res, X_test, y_train, y_train_res, y_test

def train_logistic_regression(X_train_res: pd.DataFrame, y_train_res: pd.Series, random_state: int = 42) -> LogisticRegression:
    """Treina o modelo de Regressão Logística (Baseline)."""
    print("\n[1/3] Treinando Regressão Logística...")
    model_lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=random_state
    )
    model_lr.fit(X_train_res, y_train_res)
    return model_lr

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestClassifier:
    """Treina o modelo Random Forest com ajuste de pesos de classe."""
    print("[2/3] Treinando Random Forest Classifier...")
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=random_state
    )
    # Random Forest lida nativamente bem com pesos, treinando na base original estratificada
    model_rf.fit(X_train, y_train)
    return model_rf

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> XGBClassifier:
    """Treina o modelo XGBoost com escala de peso para classes positivas."""
    print("[3/3] Treinando XGBoost Classifier...")
    
    # Razão entre negativos e positivos (ajuste para desbalanceamento)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train)
    return model_xgb

def save_artifacts(models_dict: dict, splits_dict: dict, output_dir: str = 'models'):
    """Salva modelos e conjuntos de teste em disco."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"-> Modelo salvo: {path}")
        
    for name, data in splits_dict.items():
        path = os.path.join(output_dir, f"{name}.parquet")
        if isinstance(data, pd.Series):
            data.to_frame().to_parquet(path)
        else:
            data.to_parquet(path)
        print(f"-> Partição salva: {path}")

def run_training_pipeline():
    # 1. Carregar dados
    X, y, feature_cols = load_processed_data()
    
    # 2. Particionar e balancear
    X_train, X_train_res, X_test, y_train, y_train_res, y_test = split_and_balance_data(X, y)
    
    # 3. Treinar modelos
    model_lr = train_logistic_regression(X_train_res, y_train_res)
    model_rf = train_random_forest(X_train, y_train)
    model_xgb = train_xgboost(X_train, y_train)
    
    # 4. Salvar modelos e dados de teste
    models_to_save = {
        'model_logistic_regression': model_lr,
        'model_random_forest': model_rf,
        'model_xgboost': model_xgb
    }
    
    splits_to_save = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
    print("\nSalvando artefatos gerados...")
    save_artifacts(models_to_save, splits_to_save)
    print("\n✓ Treinamento concluído com sucesso!")

if __name__ == '__main__':
    run_training_pipeline()