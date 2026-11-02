import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    brier_score_loss,
)


def read_input_data(input_path: str) -> pd.DataFrame:
    """Lê arquivos CSV identificando automaticamente separador e codificação."""
    encodings = ['utf-8-sig', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            return pd.read_csv(input_path, sep=None, engine='python', encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Não foi possível ler o arquivo: {input_path}")


def rename_and_remap(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia as colunas X1 a X18 para as nomenclaturas contábeis padronizadas."""
    summary = {
        'X1': 'current_assets',
        'X2': 'cogs',
        'X3': 'DA',
        'X4': 'EBITDA',
        'X5': 'inventory',
        'X6': 'net_income',
        'X7': 'total_receivables',
        'X8': 'market_value',
        'X9': 'net_sales',
        'X10': 'total_assets',
        'X11': 'long_term_debt',
        'X12': 'EBIT',
        'X13': 'gross_profit',
        'X14': 'current_liabilities',
        'X15': 'retained_earnings',
        'X16': 'total_revenue',
        'X17': 'total_liabilities',
        'X18': 'total_opex',
    }
    return df.rename(columns=summary)


def create_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as razões financeiras adimensionais em ambas as bases."""
    data = pd.DataFrame(index=df.index)

    # Identificadores e Alvo
    data['company_name'] = df['company_name']
    data['year'] = df['year']
    data['target'] = (df['status_label'] == 'failed').astype(int)

    # 1. Indicadores de Liquidez
    data['current_liquidity'] = df['current_assets'] / df['current_liabilities']
    data['dry_liquidity'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
    data['working_capital_over_total_assets'] = (df['current_assets'] - df['current_liabilities']) / df['total_assets']

    # 2. Indicadores de Rentabilidade e Desempenho
    data['ROA'] = df['net_income'] / df['total_assets']
    equity = df['total_assets'] - df['total_liabilities']
    data['ROE'] = np.where(equity != 0, df['net_income'] / equity, np.nan)
    data['EBIT_to_assets'] = df['EBIT'] / df['total_assets']
    data['EBITDA_to_assets'] = df['EBITDA'] / df['total_assets']
    data['retained_earnings_to_assets'] = df['retained_earnings'] / df['total_assets']
    data['net_margin'] = df['net_income'] / df['net_sales']
    data['gross_margin'] = df['gross_profit'] / df['net_sales']

    # 3. Indicadores de Endividamento e Estrutura de Capital
    data['debt_to_assets'] = df['total_liabilities'] / df['total_assets']
    data['current_debt_to_assets'] = df['current_liabilities'] / df['total_assets']
    data['long_term_debt_to_assets'] = df['long_term_debt'] / df['total_assets']
    data['equity_to_liabilities'] = (df['total_assets'] - df['total_liabilities']) / df['total_liabilities']

    # Nota: market_value existe no dataset US, mas é NaN no BR. Mantemos o rácio para compatibilização estrutural
    if 'market_value' in df.columns:
        data['market_val_to_liabilities'] = df['market_value'] / df['total_liabilities']
    else:
        data['market_val_to_liabilities'] = np.nan

    # 4. Indicadores de Eficiência e Atividade
    data['asset_turnover'] = df['net_sales'] / df['total_assets']
    data['receivables_to_sales'] = df['total_receivables'] / df['net_sales']
    data['inventory_to_sales'] = df['inventory'] / df['net_sales']
    data['cogs_to_sales'] = df['cogs'] / df['net_sales']
    data['opex_to_sales'] = df['total_opex'] / df['net_sales']

    # Substituição de infinitos por NaN
    feature_columns = data.columns.difference(['company_name', 'year', 'target'])
    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    data['financial_data_missing'] = data[feature_columns].isna().all(axis=1).astype(int)

    return data


def preprocess_cross_market(X_train_us: pd.DataFrame, X_test_br: pd.DataFrame):
    """
    Ajusta (fit) a limpeza, imputação e normalização na base Americana (Treino)
    e aplica (transform) na base Brasileira (Teste) sem vazamento de dados.
    """
    X_tr = X_train_us.replace([np.inf, -np.inf], np.nan).copy()
    X_te = X_test_br.replace([np.inf, -np.inf], np.nan).copy()

    # Garantir alinhamento estrito das colunas
    common_cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[common_cols]
    X_te = X_te[common_cols]

    # Winsorização nos percentis 1% e 99% baseada nos limites dos EUA
    lower_limits = X_tr.quantile(0.01)
    upper_limits = X_tr.quantile(0.99)
    X_tr = X_tr.clip(lower=lower_limits, upper=upper_limits, axis=1)
    X_te = X_te.clip(lower=lower_limits, upper=upper_limits, axis=1)

    # Imputação pela mediana com indicador de ausência
    imputer = SimpleImputer(strategy='median', add_indicator=True)
    tr_vals = imputer.fit_transform(X_tr)
    te_vals = imputer.transform(X_te)
    feature_names = imputer.get_feature_names_out(X_tr.columns)

    # Escalonamento Robusto
    scaler = RobustScaler()
    tr_vals = scaler.fit_transform(tr_vals)
    te_vals = scaler.transform(te_vals)

    X_train_processed = pd.DataFrame(tr_vals, index=X_tr.index, columns=feature_names)
    X_test_processed = pd.DataFrame(te_vals, index=X_te.index, columns=feature_names)

    return X_train_processed, X_test_processed


def run_cross_market_experiment(
    us_path: str = 'american_bankruptcy.csv',
    br_path: str = 'brazil_bankruptcy.csv',
    output_dir: str = 'results_cross_market',
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 75)
    print("      INICIANDO PIPELINE DE TRANSFERIBILIDADE CROSS-MARKET (EUA -> BR)")
    print("=" * 75)

    # 1. Carregamento e Feature Engineering
    print("\n1. Ingestão e cálculo de rácios contábeis...")
    df_us_raw = read_input_data(us_path)
    df_br_raw = read_input_data(br_path)

    df_us_feat = create_attributes(rename_and_remap(df_us_raw))
    df_br_feat = create_attributes(rename_and_remap(df_br_raw))

    meta_cols = ['company_name', 'year', 'target']
    feature_cols = [c for c in df_us_feat.columns if c not in meta_cols]

    # Base de treino = Partição de 70% ou 100% dos EUA. Usaremos a partição estratificada de treino (70%)
    X_us_train, _, y_us_train, _ = train_test_split(
        df_us_feat[feature_cols],
        df_us_feat['target'],
        test_size=0.30,
        random_state=42,
        stratify=df_us_feat['target'],
    )

    # Base de teste da transferibilidade = 100% da base Brasileira rotulada
    X_br_test = df_br_feat[feature_cols].copy()
    y_br_test = df_br_feat['target'].copy()

    print(f"   -> Amostras de Treino (EUA): {X_us_train.shape[0]} ({y_us_train.sum()} falências)")
    print(f"   -> Amostras de Teste (Brasil): {X_br_test.shape[0]} ({y_br_test.sum()} falências)")

    # 2. Pré-processamento sem vazamento de dados
    print("\n2. Ajustando normalização e imputação nos EUA e aplicando no Brasil...")
    X_train_p, X_test_p = preprocess_cross_market(X_us_train, X_br_test)

    # 3. Balanceamento no Treino com SMOTE
    print("\n3. Aplicando SMOTE exclusivamente no treino dos EUA...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_p, y_us_train)

    # 4. Treinamento dos Modelos nos Dados dos EUA
    print("\n4. Treinando modelos com os padrões norte-americanos...")
    # A. Regressão Logística
    model_lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    model_lr.fit(X_train_res, y_train_res)

    # B. Random Forest
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
    )
    model_rf.fit(X_train_p, y_us_train)

    # C. XGBoost
    pos_weight = (y_us_train == 0).sum() / (y_us_train == 1).sum()
    model_xgb = XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        gamma=1,
        reg_lambda=1,
        reg_alpha=1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    model_xgb.fit(X_train_p, y_us_train)

    models = {
        'Regressão Logística': model_lr,
        'Random Forest': model_rf,
        'XGBoost': model_xgb,
    }

    # Salvar modelos treinados cross-market
    for name, m in models.items():
        joblib.dump(m, f"models/cross_market_{name.lower().replace(' ', '_')}.joblib")

    # 5. Avaliação Direta na Base Brasileira
    print("\n5. Submetendo as empresas brasileiras aos modelos norte-americanos...")
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    plt.figure(figsize=(8, 6), dpi=300)

    for idx, (name, m) in enumerate(models.items()):
        y_prob = m.predict_proba(X_test_p)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_br_test, y_pred)
        prec = precision_score(y_br_test, y_pred, zero_division=0)
        rec = recall_score(y_br_test, y_pred, zero_division=0)
        f1 = f1_score(y_br_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_br_test, y_prob)
        brier = brier_score_loss(y_br_test, y_prob)
        cm = confusion_matrix(y_br_test, y_pred)

        results.append({
            'Modelo': name,
            'Acurácia': round(acc, 4),
            'Precisão': round(prec, 4),
            'Recall (Sensibilidade)': round(rec, 4),
            'F1-Score': round(f1, 4),
            'AUC-ROC': round(auc, 4),
            'Brier Score': round(brier, 4),
            'Falências Detectadas (TP)': f"{cm[1, 1]}/{cm[1, 0] + cm[1, 1]}",
            'Alarmes Falsos (FP)': cm[0, 1],
        })

        # Matriz de Confusão individual
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=axes[idx],
            xticklabels=['Ativa (0)', 'Falência (1)'],
            yticklabels=['Ativa (0)', 'Falência (1)'],
            annot_kws={'size': 11, 'weight': 'bold'},
        )
        axes[idx].set_title(f'Transferibilidade: {name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predição (Modelo EUA)', fontsize=10)
        axes[idx].set_ylabel('Rótulo Real (Brasil)', fontsize=10)

        # Plot Curva ROC
        fpr, tpr, _ = roc_curve(y_br_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})')

    # Salvar gráficos
    axes[0].figure.tight_layout()
    axes[0].figure.savefig(os.path.join(output_dir, 'cross_market_confusion_matrices.png'))
    plt.close(axes[0].figure)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Classificador Aleatório')
    plt.title('Curvas ROC - Transferibilidade Cross-Market (EUA -> Brasil)', fontweight='bold')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_market_roc_curves.png'))
    plt.close()

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'cross_market_metrics.csv'), index=False)

    print("\n" + "=" * 80)
    print("           TABELA DE RESULTADOS DA TRANSFERIBILIDADE (EUA -> BRASIL)")
    print("=" * 80)
    print(df_results.to_markdown(index=False))
    print("=" * 80)

    # 6. Diagnóstico de Transferibilidade (Positivo vs Negativo)
    best_auc = df_results['AUC-ROC'].max()
    best_model = df_results.loc[df_results['AUC-ROC'].idxmax(), 'Modelo']

    print("\n" + "#" * 80)
    print("                      VEREDITO METODOLÓGICO")
    print("#" * 80)
    if best_auc >= 0.75:
        print(f"-> RESULTADO: TRANSFERIBILIDADE POSITIVA E ROBUSTA!")
        print(f"   O modelo '{best_model}' alcançou AUC-ROC de {best_auc:.4f} no Brasil sem nunca ter")
        print("   visto dados brasileiros no treinamento. Isso comprova que a criação de rácios")
        print("   adimensionais permitiu que os mecanismos financeiros de default fossem generalizados.")
    elif best_auc >= 0.65:
        print(f"-> RESULTADO: TRANSFERIBILIDADE MODERADA (AUC = {best_auc:.4f}).")
        print("   O modelo detecta a ordem de risco, mas sofre com disparidades estruturais (ex: Custo Brasil).")
    else:
        print(f"-> RESULTADO: TRANSFERIBILIDADE NEGATIVA/FRACA (AUC = {best_auc:.4f}).")
        print("   O modelo treinado nos EUA não conseguiu distinguir falências no Brasil de forma confiável.")
    print("#" * 80 + "\n")


if __name__ == '__main__':
    run_cross_market_experiment(
        us_path='american_bankruptcy.csv',
        br_path='brazil_bankruptcy.csv',
        output_dir='results_cross_market',
    )