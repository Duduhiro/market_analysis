from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BRAZIL_DIR = BASE_DIR / "brazil_database"
INPUT_COMPANIES_PATH = BASE_DIR / "empresas_painel_2021_2025.csv"
OUTPUT_PANEL_PATH = BASE_DIR / "brazilian_financial_panel_2021_2025.csv"

YEARS = [2021, 2022, 2023, 2024, 2025]
STATEMENTS = ["BPA", "BPP", "DRE", "DVA"]
KEY_COLS = ["CNPJ_CIA", "ANO"]
X_COLS = [f"X{i}" for i in range(1, 19)]
ULTIMO = "\u00daLTIMO"


def statement_path(year: int, statement: str) -> Path:
    return BRAZIL_DIR / str(year) / f"dfp_cia_aberta_{statement}_con_{year}.csv"


def read_cvm_csv(path: Path, usecols: set[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype={"CNPJ_CIA": "string", "CD_CONTA": "string"},
        usecols=lambda col: col in usecols,
    )
    df.columns = [col.strip() for col in df.columns]
    if "CNPJ_CIA" in df.columns:
        df["CNPJ_CIA"] = df["CNPJ_CIA"].astype("string").str.strip()
    return df


def read_target_companies() -> pd.DataFrame:
    if not INPUT_COMPANIES_PATH.exists():
        raise FileNotFoundError(f"Arquivo de empresas nao encontrado: {INPUT_COMPANIES_PATH}")

    companies = pd.read_csv(
        INPUT_COMPANIES_PATH,
        dtype={"CNPJ_CIA": "string", "CD_CVM": "string", "DENOM_CIA": "string"},
    )
    expected_cols = {"CNPJ_CIA", "CD_CVM", "DENOM_CIA"}
    missing_cols = expected_cols - set(companies.columns)
    if missing_cols:
        raise ValueError(f"Colunas ausentes em {INPUT_COMPANIES_PATH}: {sorted(missing_cols)}")

    companies = companies[["CNPJ_CIA", "CD_CVM", "DENOM_CIA"]].copy()
    companies["CNPJ_CIA"] = companies["CNPJ_CIA"].astype("string").str.strip()
    companies["CD_CVM"] = companies["CD_CVM"].astype("string").str.strip()
    companies["DENOM_CIA"] = companies["DENOM_CIA"].astype("string").str.strip()

    duplicate_count = int(companies["CNPJ_CIA"].duplicated().sum())
    if duplicate_count:
        duplicates = companies.loc[companies["CNPJ_CIA"].duplicated(keep=False)].sort_values("CNPJ_CIA")
        raise ValueError(
            "Arquivo de empresas contem CNPJ_CIA duplicado:\n"
            + duplicates.to_string(index=False)
        )

    return companies.sort_values("CNPJ_CIA").reset_index(drop=True)


def apply_common_statement_filters(df: pd.DataFrame, year: int, allowed_cnpjs: set[str]) -> pd.DataFrame:
    filtered = df.loc[df["CNPJ_CIA"].isin(allowed_cnpjs)].copy()

    if "ORDEM_EXERC" in filtered.columns:
        filtered = filtered.loc[
            filtered["ORDEM_EXERC"].astype("string").str.strip().eq(ULTIMO)
        ].copy()

    parsed_dt = pd.to_datetime(filtered["DT_REFER"], errors="coerce")
    invalid_dates = filtered.loc[parsed_dt.isna() & filtered["DT_REFER"].notna()]
    if not invalid_dates.empty:
        raise ValueError(
            f"DT_REFER invalido encontrado no ano {year}:\n"
            + invalid_dates[["CNPJ_CIA", "DT_REFER"]].drop_duplicates().head(20).to_string(index=False)
        )

    filtered = filtered.loc[parsed_dt.dt.year.eq(year)].copy()
    filtered["_DT_REFER_SORT"] = pd.to_datetime(filtered["DT_REFER"], errors="coerce")

    if "VERSAO" in filtered.columns:
        filtered["_VERSAO_NUM"] = pd.to_numeric(filtered["VERSAO"], errors="coerce")
        latest = filtered.groupby(["CNPJ_CIA", "DT_REFER"], dropna=False)["_VERSAO_NUM"].transform("max")
        filtered = filtered.loc[filtered["_VERSAO_NUM"].eq(latest)].copy()

    return filtered


def statement_dt_refer(year: int, statement: str, allowed_cnpjs: set[str]) -> pd.DataFrame:
    path = statement_path(year, statement)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = read_cvm_csv(
        path,
        usecols={"CNPJ_CIA", "DT_REFER", "ORDEM_EXERC", "VERSAO"},
    )
    filtered = apply_common_statement_filters(df, year, allowed_cnpjs)
    return filtered[["CNPJ_CIA", "DT_REFER", "_DT_REFER_SORT"]].drop_duplicates()


def resolve_dt_refer_for_year(year: int, companies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed_cnpjs = set(companies["CNPJ_CIA"])
    statement_dates = {
        statement: statement_dt_refer(year, statement, allowed_cnpjs)
        for statement in STATEMENTS
    }

    missing_rows = []
    no_common_rows = []
    multi_common_rows = []
    resolved_rows = []

    for cnpj in sorted(allowed_cnpjs):
        date_sets = {}
        for statement, dates in statement_dates.items():
            cnpj_dates = dates.loc[dates["CNPJ_CIA"].eq(cnpj), "DT_REFER"].dropna().astype(str)
            date_sets[statement] = set(cnpj_dates)
            if not date_sets[statement]:
                missing_rows.append({"CNPJ_CIA": cnpj, "ANO": year, "DEMONSTRACAO": statement})

        if any(not dates for dates in date_sets.values()):
            continue

        common_dates = set.intersection(*(date_sets[statement] for statement in STATEMENTS))
        if not common_dates:
            row = {"CNPJ_CIA": cnpj, "ANO": year}
            row.update({statement: ",".join(sorted(date_sets[statement])) for statement in STATEMENTS})
            no_common_rows.append(row)
            continue

        chosen_dt = max(common_dates, key=lambda value: pd.Timestamp(value))
        if len(common_dates) > 1:
            multi_common_rows.append(
                {
                    "CNPJ_CIA": cnpj,
                    "ANO": year,
                    "DT_REFER_ESCOLHIDO": chosen_dt,
                    "DT_REFER_COMUNS": ",".join(sorted(common_dates)),
                }
            )

        resolved_rows.append({"CNPJ_CIA": cnpj, "ANO": year, "DT_REFER": chosen_dt})

    missing = pd.DataFrame(missing_rows)
    no_common = pd.DataFrame(no_common_rows)
    multi_common = pd.DataFrame(multi_common_rows)

    if not missing.empty:
        raise ValueError(
            f"Empresas do painel sem DT_REFER em demonstracoes obrigatorias no ano {year}:\n"
            + missing.to_string(index=False)
        )

    if not no_common.empty:
        raise ValueError(
            f"Empresas sem DT_REFER comum entre BPA/BPP/DRE/DVA no ano {year}:\n"
            + no_common.to_string(index=False)
        )

    resolved = pd.DataFrame(resolved_rows)
    return resolved, multi_common, pd.concat(
        [
            dates.assign(ANO=year, DEMONSTRACAO=statement)
            for statement, dates in statement_dates.items()
        ],
        ignore_index=True,
    )


def build_panel_skeleton(companies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    skeleton = companies.merge(pd.DataFrame({"ANO": YEARS}), how="cross")
    dt_frames = []
    multi_dt_frames = []

    for year in YEARS:
        year_dt, multi_common, _statement_dates = resolve_dt_refer_for_year(year, companies)
        dt_frames.append(year_dt)
        if not multi_common.empty:
            multi_dt_frames.append(multi_common)

    dt_refer = pd.concat(dt_frames, ignore_index=True)
    panel = skeleton.merge(dt_refer, on=KEY_COLS, how="left")

    for col in X_COLS:
        panel[col] = np.nan

    ordered_cols = ["CNPJ_CIA", "CD_CVM", "DENOM_CIA", "ANO", "DT_REFER", *X_COLS]
    panel = panel[ordered_cols].sort_values(KEY_COLS).reset_index(drop=True)
    multi_dt = pd.concat(multi_dt_frames, ignore_index=True) if multi_dt_frames else pd.DataFrame()
    return panel, multi_dt


def validate_panel(panel: pd.DataFrame, companies: pd.DataFrame) -> dict[str, int]:
    expected_companies = set(companies["CNPJ_CIA"])
    panel_companies = set(panel["CNPJ_CIA"])
    outside_companies = panel_companies - expected_companies
    missing_companies = expected_companies - panel_companies

    if outside_companies:
        raise ValueError(f"Painel contem empresas fora do universo: {sorted(outside_companies)}")
    if missing_companies:
        raise ValueError(f"Painel nao contem todas as empresas esperadas: {sorted(missing_companies)}")

    duplicated_key_count = int(panel.duplicated(KEY_COLS).sum())
    if duplicated_key_count:
        duplicates = panel.loc[panel.duplicated(KEY_COLS, keep=False), KEY_COLS + ["DENOM_CIA"]]
        raise ValueError(
            "Duplicidades CNPJ_CIA + ANO encontradas:\n"
            + duplicates.sort_values(KEY_COLS).to_string(index=False)
        )

    expected_years = set(YEARS)
    observations_by_cnpj = panel.groupby("CNPJ_CIA")["ANO"].agg(["size", lambda values: set(values)])
    observations_by_cnpj.columns = ["QTD_OBSERVACOES", "ANOS"]

    less_than_five = int(observations_by_cnpj["QTD_OBSERVACOES"].lt(5).sum())
    exactly_five = int(observations_by_cnpj["QTD_OBSERVACOES"].eq(5).sum())
    wrong_years = observations_by_cnpj.loc[observations_by_cnpj["ANOS"].ne(expected_years)]

    if not wrong_years.empty:
        raise ValueError(
            "CNPJs sem exatamente os anos 2021-2025:\n"
            + wrong_years.head(20).to_string()
        )

    if panel["DT_REFER"].isna().any():
        missing_dt = panel.loc[panel["DT_REFER"].isna(), KEY_COLS + ["DENOM_CIA"]]
        raise ValueError(
            "Linhas sem DT_REFER no painel:\n"
            + missing_dt.to_string(index=False)
        )

    return {
        "numero_empresas": len(expected_companies),
        "observacoes_esperadas": len(expected_companies) * len(YEARS),
        "observacoes_reais": len(panel),
        "cnpjs_exatamente_5_observacoes": exactly_five,
        "cnpjs_menos_de_5_observacoes": less_than_five,
        "duplicidades_cnpj_ano": duplicated_key_count,
    }


def print_report(panel: pd.DataFrame, companies: pd.DataFrame, metrics: dict[str, int], multi_dt: pd.DataFrame) -> None:
    print("RELATORIO - ESQUELETO DO PAINEL BRASILEIRO 2021-2025")
    print()
    print(f"arquivo_empresas_utilizado: {INPUT_COMPANIES_PATH}")
    print(f"arquivo_painel_gerado: {OUTPUT_PANEL_PATH}")
    print()
    print(f"Numero de empresas: {metrics['numero_empresas']}")
    print(f"Numero esperado de observacoes: {metrics['observacoes_esperadas']}")
    print(f"Numero real de observacoes: {metrics['observacoes_reais']}")
    print(f"CNPJs com exatamente 5 observacoes: {metrics['cnpjs_exatamente_5_observacoes']}")
    print(f"CNPJs com menos de 5 observacoes: {metrics['cnpjs_menos_de_5_observacoes']}")
    print(f"Duplicidades CNPJ_CIA + ANO: {metrics['duplicidades_cnpj_ano']}")
    print(f"Empresas fora do arquivo de controle: {len(set(panel['CNPJ_CIA']) - set(companies['CNPJ_CIA']))}")
    print()
    print("Confirmacao: todas as empresas possuem exatamente cinco anos: OK")
    print("Confirmacao: nao existem duplicidades CNPJ_CIA + ANO: OK")
    print("Confirmacao: variaveis X1-X18 inicializadas como NaN, sem calculo financeiro: OK")
    print()
    print(f"CNPJs/anos com mais de um DT_REFER comum nas quatro demonstracoes: {multi_dt.shape[0]}")
    if not multi_dt.empty:
        print("Nesses casos foi usado o DT_REFER mais recente do ano para manter uma linha por CNPJ_CIA + ANO:")
        print(multi_dt.to_string(index=False))
        print()
    print("Primeiras 20 linhas:")
    print(panel.head(20).to_string(index=False))
    print()
    print("Ultimas 20 linhas:")
    print(panel.tail(20).to_string(index=False))


def main() -> None:
    companies = read_target_companies()
    panel, multi_dt = build_panel_skeleton(companies)
    metrics = validate_panel(panel, companies)

    panel.to_csv(OUTPUT_PANEL_PATH, sep=";", index=False, encoding="utf-8-sig")
    print_report(panel, companies, metrics, multi_dt)


if __name__ == "__main__":
    main()
