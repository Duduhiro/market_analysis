import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
    classification_report
)

def load_evaluation_artifacts(models_dir: str = 'models'):
    """Carrega os modelos treinados e os dados de teste salvos."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Diretório {models_dir} não encontrado. Execute o Módulo 2 primeiro.")
    
    print("-> A carregar conjuntos de teste...")
    X_test = pd.read_parquet(os.path.join(models_dir, 'X_test.parquet'))
    y_test_df = pd.read_parquet(os.path.join(models_dir, 'y_test.parquet'))
    y_test = y_test_df.iloc[:, 0]
    
    print("-> A carregar modelos treinados...")
    models = {
        'Regressão Logística': joblib.load(os.path.join(models_dir, 'model_logistic_regression.joblib')),
        'Random Forest': joblib.load(os.path.join(models_dir, 'model_random_forest.joblib')),
        'XGBoost': joblib.load(os.path.join(models_dir, 'model_xgboost.joblib'))
    }
    
    return models, X_test, y_test

def compute_metrics_benchmark(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Calcula as métricas consolidadas de desempenho para todos os modelos."""
    records = []
    
    for name, model in models.items():
        # Predições de classe e probabilidade
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        records.append({
            'Modelo': name,
            'Acurácia': round(acc, 4),
            'Precisão': round(prec, 4),
            'Recall (Sensibilidade)': round(rec, 4),
            'F1-Score': round(f1, 4),
            'AUC-ROC': round(auc, 4),
            'Brier Score': round(brier, 4)
        })
        
    df_metrics = pd.DataFrame(records)
    return df_metrics

def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: str):
    """Gera o gráfico comparativo das curvas ROC."""
    plt.figure(figsize=(8, 6), dpi=300)
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Classificador Aleatório')
    plt.title('Comparativo de Curvas ROC - Previsão de Falência', fontsize=12, fontweight='bold')
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=10)
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"-> Curva ROC salva em: {output_path}")

def plot_confusion_matrices(models: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: str):
    """Gera e salva matrizes de confusão para os 3 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
            xticklabels=['Ativa (0)', 'Falência (1)'],
            yticklabels=['Ativa (0)', 'Falência (1)'],
            annot_kws={'size': 11, 'weight': 'bold'}
        )
        axes[idx].set_title(f'Matriz: {name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predição do Modelo', fontsize=10)
        axes[idx].set_ylabel('Rótulo Real', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"-> Matrizes de Confusão salvas em: {output_path}")

def plot_feature_importances(models: dict, feature_names: list, output_path: str):
    """Gera o ranking de importância dos rácios financeiros para Random Forest e XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    tree_models = [('Random Forest', models['Random Forest']), ('XGBoost', models['XGBoost'])]
    
    for idx, (name, model) in enumerate(tree_models):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:12] # Top 12 variáveis
            
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            sns.barplot(x=top_importances, y=top_features, ax=axes[idx], palette='viridis')
            axes[idx].set_title(f'Top Atributos - {name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Importância Relativa', fontsize=10)
            axes[idx].grid(True, linestyle=':', alpha=0.6)
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"-> Feature Importance salva em: {output_path}")

def generate_shap_analysis(model_xgb, X_test: pd.DataFrame, output_path: str):
    """Calcula e gera a análise de interpretabilidade SHAP para o modelo XGBoost."""
    try:
        import shap
        print("-> A calcular valores SHAP (amostra de 500 observações)...")
        # Subamostra para cálculo ágil
        X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('Impacto SHAP dos Indicadores Contábeis (XGBoost)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"-> Análise SHAP salva em: {output_path}")
    except ImportError:
        print("! Biblioteca 'shap' não instalada. Execute 'pip install shap' para gerar este gráfico.")

def run_evaluation_pipeline(output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carregar modelos e dados
    models, X_test, y_test = load_evaluation_artifacts()
    feature_names = X_test.columns.tolist()
    
    # 2. Gerar tabela de métricas
    print("\nCalculando métricas do Benchmark...")
    df_metrics = compute_metrics_benchmark(models, X_test, y_test)
    
    metrics_csv = os.path.join(output_dir, 'benchmark_metrics.csv')
    df_metrics.to_csv(metrics_csv, index=False)
    
    print("\n" + "="*70)
    print("                  TABELA CONSOLIDADA DO BENCHMARK")
    print("="*70)
    print(df_metrics.to_markdown(index=False))
    print("="*70)
    
    # 3. Gerar gráficos
    print("\nA gerar gráficos de avaliação...")
    plot_roc_curves(models, X_test, y_test, os.path.join(output_dir, 'roc_curve_comparison.png'))
    plot_confusion_matrices(models, X_test, y_test, os.path.join(output_dir, 'confusion_matrices.png'))
    plot_feature_importances(models, feature_names, os.path.join(output_dir, 'feature_importance.png'))
    generate_shap_analysis(models['XGBoost'], X_test, os.path.join(output_dir, 'shap_summary.png'))
    
    print(f"\n✓ Todos os resultados foram exportados para o diretório '{output_dir}/'.")

if __name__ == '__main__':
    run_evaluation_pipeline()