import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def rename_and_remap(df: pd.DataFrame) -> pd.DataFrame:

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
    
    df = df.rename(columns=summary)
    
    return df

def create_attributes(df: pd.DataFrame) -> pd.DataFrame:

    data = pd.DataFrame(index=df.index)

    # Metadados e Alvo
    data['company_name'] = df['company_name']
    data['year'] = df['year']
    data['target'] = (df['status_label'] == 'failed').astype(int)

    # calculando indicadores financeiros

    data['current_liquidity'] = df['current_assets'] / df['current_liabilities']
    data['dry_liquidity'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
    data['working_capital_over_total_assets'] = (df['current_assets'] - df['current_liabilities']) / df['total_assets']

    # calculando indicadores de rentabilidade e desempenho

    data['ROA'] = df['net_income'] / df['total_assets']
    equity = df['total_assets'] - df['total_liabilities']
    data['ROE'] = np.where(equity != 0, df['net_income'] / equity, np.nan)
    data['EBIT_to_assets'] = df['EBIT'] / df['total_assets']
    data['EBITDA_to_assets'] = df['EBITDA'] / df['total_assets']
    data['retained_earnings_to_assets'] = df['retained_earnings'] / df['total_assets']
    data['net_margin'] = df['net_income'] / df['net_sales']
    data['gross_margin'] = df['gross_profit'] / df['net_sales']

    # calculando indicadores de endividamento e estrutura de capital

    data['debt_to_assets'] = df['total_liabilities'] / df['total_assets']
    data['current_debt_to_assets'] = df['current_liabilities'] / df['total_assets']
    data['long_term_debt_to_assets'] = df['long_term_debt'] / df['total_assets']
    data['market_val_to_liabilities'] = df['market_value'] / df['total_liabilities']
    data['equity_to_liabilities'] = (df['total_assets'] - df['total_liabilities']) / df['total_liabilities']

    # calculando indicadores de eficiencia operacional e atividade

    data['asset_turnover'] = df['net_sales'] / df['total_assets']
    data['receivables_to_sales'] = df['total_receivables'] / df['net_sales']
    data['inventory_to_sales'] = df['inventory'] / df['net_sales']
    data['cogs_to_sales'] = df['cogs'] / df['net_sales']
    data['opex_to_sales'] = df['total_opex'] / df['net_sales']

    return data

def normalize_attributes(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:

    df_clean = df.copy()

    # substituindo valores infinitos por NaN
    df_clean[feature_columns] = df_clean[feature_columns].replace([np.inf, -np.inf], np.nan)

    for col in feature_columns:
        
        # inputando valores nulos com a mediana da coluna
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

        # truncando valores extremos usando os percentis 1 e 99
        lower_limit = df_clean[col].quantile(0.01)
        upper_limit = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(lower=lower_limit, upper=upper_limit)

    return df_clean

def run_pipeline(
    input_path: str = 'american_bankruptcy.csv',
    output_path: str = 'processed_data.parquet',
):
    print("1. Carregando dados brutos")
    raw_df = pd.read_csv(input_path)

    print("2. Renomeando e remapeando colunas")
    df = rename_and_remap(raw_df)
    df = create_attributes(df)

    print("3. Normalizando atributos")
    feature_columns = df.columns.difference(['company_name', 'year', 'target']).tolist()
    df = normalize_attributes(df, feature_columns)

    print(f"4. Salvando dados processados em {output_path}")
    
    scaler = RobustScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    df.to_parquet(output_path, index=False)
    joblib.dump(scaler, 'robust_scaler.joblib')

    print("Pipeline de processamento concluído com sucesso!")


if __name__ == "__main__":

    run_pipeline()