from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BRAZIL_DIR = BASE_DIR / "brazil_database"
PANEL_PATH = BASE_DIR / "brazilian_financial_panel_2021_2025.csv"

YEARS = [2021, 2022, 2023, 2024, 2025]
KEY_COLS = ["CNPJ_CIA", "ANO"]
X_COLS = [f"X{i}" for i in range(1, 19)]
PANEL_COLS = ["CNPJ_CIA", "CD_CVM", "DENOM_CIA", "ANO", "DT_REFER", *X_COLS]
ULTIMO = "\u00daLTIMO"

DESC = {
    "ativo_circulante": "Ativo Circulante",
    "cogs": "Custo dos Bens e/ou Servi\u00e7os Vendidos",
    "da": "Deprecia\u00e7\u00e3o, Amortiza\u00e7\u00e3o e Exaust\u00e3o",
    "estoques": "Estoques",
    "net_income": "Lucro/Preju\u00edzo Consolidado do Per\u00edodo",
    "receivables": "Contas a Receber",
    "net_sales": "Receita de Venda de Bens e/ou Servi\u00e7os",
    "total_assets": "Ativo Total",
    "long_term_debt": "Empr\u00e9stimos e Financiamentos",
    "ebit": "Resultado Antes do Resultado Financeiro e dos Tributos",
    "resultado_bruto": "Resultado Bruto",
    "current_liabilities": "Passivo Circulante",
    "reservas_lucros": "Reservas de Lucros",
    "lucros_acumulados": "Lucros/Preju\u00edzos Acumulados",
    "passivo_nao_circulante": "Passivo N\u00e3o Circulante",
    "patrimonio_liquido": "Patrim\u00f4nio L\u00edquido Consolidado",
}

ACCOUNT_SPECS = [
    {"var": "X1", "statement": "BPA", "code": "1.01", "desc": DESC["ativo_circulante"], "abs": False},
    {"var": "X2", "statement": "DRE", "code": "3.02", "desc": DESC["cogs"], "abs": True},
    {"var": "X3", "statement": "DVA", "code": "7.04.01", "desc": DESC["da"], "abs": True},
    {"var": "X5", "statement": "BPA", "code": "1.01.04", "desc": DESC["estoques"], "abs": False},
    {"var": "X6", "statement": "DRE", "code": "3.11", "desc": DESC["net_income"], "abs": False},
    {"var": "X7", "statement": "BPA", "code": "1.01.03", "desc": DESC["receivables"], "abs": False},
    {"var": "X9", "statement": "DRE", "code": "3.01", "desc": DESC["net_sales"], "abs": False},
    {"var": "X10", "statement": "BPA", "code": "1", "desc": DESC["total_assets"], "abs": False},
    {"var": "X11", "statement": "BPP", "code": "2.02.01", "desc": DESC["long_term_debt"], "abs": False},
    {"var": "X12", "statement": "DRE", "code": "3.05", "desc": DESC["ebit"], "abs": False},
    {"var": "X14", "statement": "BPP", "code": "2.01", "desc": DESC["current_liabilities"], "abs": False},
]


def statement_path(year: int, statement: str) -> Path:
    return BRAZIL_DIR / str(year) / f"dfp_cia_aberta_{statement}_con_{year}.csv"


def capital_path(year: int) -> Path:
    return BRAZIL_DIR / str(year) / f"dfp_cia_aberta_composicao_capital_{year}.csv"


def read_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH, sep=";", encoding="utf-8-sig", dtype={"CNPJ_CIA": "string", "CD_CVM": "string"})
    if panel.shape[1] == 1:
        raise ValueError("Painel parece estar em uma unica coluna. Regrave-o com sep=';' antes de continuar.")

    missing_cols = [col for col in PANEL_COLS if col not in panel.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes no painel: {missing_cols}")

    panel = panel[PANEL_COLS].copy()
    panel["CNPJ_CIA"] = panel["CNPJ_CIA"].astype("string").str.strip()
    panel["CD_CVM"] = panel["CD_CVM"].astype("string").str.strip()
    panel["DENOM_CIA"] = panel["DENOM_CIA"].astype("string").str.strip()
    panel["ANO"] = pd.to_numeric(panel["ANO"], errors="raise").astype(int)
    panel["DT_REFER"] = panel["DT_REFER"].astype("string").str.strip()
    return panel.sort_values(KEY_COLS).reset_index(drop=True)


def read_cvm_statement(path: Path, statement: str) -> pd.DataFrame:
    usecols = {
        "CNPJ_CIA",
        "DENOM_CIA",
        "DT_REFER",
        "VERSAO",
        "ORDEM_EXERC",
        "CD_CONTA",
        "DS_CONTA",
        "VL_CONTA",
        "ESCALA_MOEDA",
    }
    if statement in {"DRE", "DVA"}:
        usecols.add("DT_FIM_EXERC")

    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype={"CNPJ_CIA": "string", "CD_CONTA": "string"},
        usecols=lambda col: col in usecols,
    )
    df.columns = [col.strip() for col in df.columns]
    df["CNPJ_CIA"] = df["CNPJ_CIA"].astype("string").str.strip()
    df["CD_CONTA"] = df["CD_CONTA"].astype("string").str.strip()
    return df


def read_capital(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype={"CNPJ_CIA": "string"},
    )
    df.columns = [col.strip() for col in df.columns]
    df["CNPJ_CIA"] = df["CNPJ_CIA"].astype("string").str.strip()
    return df


def normalize_currency_to_brl(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    vl_conta = pd.to_numeric(df["VL_CONTA"], errors="coerce")
    escala = df["ESCALA_MOEDA"].astype("string").str.strip().str.upper()
    unknown_scales = sorted(escala.dropna()[~escala.dropna().isin(["MIL", "UNIDADE"])].unique())

    values = pd.Series(np.nan, index=df.index, dtype="float64")
    values.loc[escala.eq("MIL")] = vl_conta.loc[escala.eq("MIL")] * 1000
    values.loc[escala.eq("UNIDADE")] = vl_conta.loc[escala.eq("UNIDADE")]
    return values, unknown_scales


def filtered_statement(year: int, statement: str, panel_keys: pd.DataFrame) -> pd.DataFrame:
    path = statement_path(year, statement)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = read_cvm_statement(path, statement)
    year_keys = panel_keys.loc[panel_keys["ANO"].eq(year), ["CNPJ_CIA", "ANO", "DT_REFER"]].copy()
    filtered = df.merge(year_keys, on=["CNPJ_CIA", "DT_REFER"], how="inner")

    filtered = filtered.loc[
        filtered["ORDEM_EXERC"].astype("string").str.strip().eq(ULTIMO)
    ].copy()

    if statement in {"DRE", "DVA"}:
        filtered = filtered.loc[
            filtered["DT_FIM_EXERC"].astype("string").str.strip().eq(
                filtered["DT_REFER"].astype("string").str.strip()
            )
        ].copy()

    filtered["VERSAO"] = pd.to_numeric(filtered["VERSAO"], errors="coerce")
    latest = filtered.groupby(["CNPJ_CIA", "ANO", "DT_REFER"], dropna=False)["VERSAO"].transform("max")
    return filtered.loc[filtered["VERSAO"].eq(latest)].copy()


def find_conflicts(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    conflict_key = ["CNPJ_CIA", "ANO", "DT_REFER", "CD_CONTA", "DS_CONTA"]
    value_cols = ["VERSAO", "ESCALA_MOEDA", "VL_CONTA", value_col]
    counts = df.groupby(conflict_key, dropna=False)[value_cols].nunique(dropna=False).reset_index()
    conflict_keys = counts.loc[counts[value_cols].gt(1).any(axis=1), conflict_key]
    if conflict_keys.empty:
        return df.iloc[0:0].copy()
    return df.merge(conflict_keys, on=conflict_key, how="inner")


def extract_account(panel: pd.DataFrame, spec: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    variable = str(spec["var"])
    statement = str(spec["statement"])
    code = str(spec["code"])
    description = str(spec["desc"])
    use_abs = bool(spec["abs"])
    panel_keys = panel[["CNPJ_CIA", "ANO", "DT_REFER"]].copy()

    values = []
    exact_duplicates = {}
    unknown_scales = {}
    conflicts = []
    different_descriptions = []
    original_signs = {}

    for year in YEARS:
        filtered = filtered_statement(year, statement, panel_keys)
        same_code = filtered.loc[filtered["CD_CONTA"].eq(code)].copy()
        diff_desc = same_code.loc[
            ~same_code["DS_CONTA"].astype("string").str.strip().eq(description)
        ].copy()
        account = same_code.loc[
            same_code["DS_CONTA"].astype("string").str.strip().eq(description)
        ].copy()

        exact_duplicates[year] = int(account.duplicated().sum())
        account = account.drop_duplicates()

        raw_values = pd.to_numeric(account["VL_CONTA"], errors="coerce")
        original_signs[year] = {
            "negativos": int(raw_values.lt(0).sum()),
            "zero": int(raw_values.eq(0).sum()),
            "positivos": int(raw_values.gt(0).sum()),
        }

        account[variable], year_unknown_scales = normalize_currency_to_brl(account)
        if use_abs:
            account[variable] = account[variable].abs()

        conflict_rows = find_conflicts(account, variable)
        if not conflict_rows.empty:
            conflict_keys = conflict_rows[["CNPJ_CIA", "ANO"]].drop_duplicates().assign(_conflict=True)
            account = account.merge(conflict_keys, on=["CNPJ_CIA", "ANO"], how="left")
            account = account.loc[account["_conflict"].isna()].drop(columns=["_conflict"])
            conflicts.append(conflict_rows)

        extracted = account[["CNPJ_CIA", "ANO", variable]].drop_duplicates()
        if extracted.duplicated(KEY_COLS).any():
            duplicated = extracted.loc[extracted.duplicated(KEY_COLS, keep=False)].sort_values(KEY_COLS)
            raise ValueError(f"Duplicidade de {variable} depois dos filtros:\n{duplicated.to_string(index=False)}")

        values.append(extracted)
        unknown_scales[year] = year_unknown_scales
        if not diff_desc.empty:
            different_descriptions.append(diff_desc)

    diagnostics = {
        "variable": variable,
        "statement": statement,
        "code": code,
        "description": description,
        "exact_duplicates": exact_duplicates,
        "unknown_scales": unknown_scales,
        "conflicts": pd.concat(conflicts, ignore_index=True) if conflicts else pd.DataFrame(),
        "different_descriptions": (
            pd.concat(different_descriptions, ignore_index=True) if different_descriptions else pd.DataFrame()
        ),
        "original_signs": original_signs,
    }
    return pd.concat(values, ignore_index=True), diagnostics


def assign_variable(panel: pd.DataFrame, values: pd.DataFrame, variable: str) -> pd.DataFrame:
    if values.duplicated(KEY_COLS).any():
        duplicates = values.loc[values.duplicated(KEY_COLS, keep=False)].sort_values(KEY_COLS)
        raise ValueError(f"Valores duplicados para {variable}:\n{duplicates.to_string(index=False)}")

    merged = panel[KEY_COLS].merge(values[KEY_COLS + [variable]], on=KEY_COLS, how="left")
    panel[variable] = pd.to_numeric(merged[variable], errors="coerce").to_numpy()
    return panel


def extract_component_accounts(
    panel: pd.DataFrame,
    statement: str,
    components: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    panel_keys = panel[["CNPJ_CIA", "ANO", "DT_REFER"]].copy()
    all_components = []
    diagnostics = {
        "exact_duplicates": {},
        "unknown_scales": {},
        "conflicts": [],
        "different_descriptions": [],
    }

    for year in YEARS:
        filtered = filtered_statement(year, statement, panel_keys)
        diagnostics["exact_duplicates"][year] = {}
        diagnostics["unknown_scales"][year] = {}

        for component_col, code, description in components:
            same_code = filtered.loc[filtered["CD_CONTA"].eq(code)].copy()
            diff_desc = same_code.loc[
                ~same_code["DS_CONTA"].astype("string").str.strip().eq(description)
            ].copy()
            account = same_code.loc[
                same_code["DS_CONTA"].astype("string").str.strip().eq(description)
            ].copy()
            diagnostics["exact_duplicates"][year][component_col] = int(account.duplicated().sum())
            account = account.drop_duplicates()
            account[component_col], unknown_scales = normalize_currency_to_brl(account)
            diagnostics["unknown_scales"][year][component_col] = unknown_scales

            conflict_rows = find_conflicts(account, component_col)
            if not conflict_rows.empty:
                conflict_keys = conflict_rows[["CNPJ_CIA", "ANO"]].drop_duplicates().assign(_conflict=True)
                account = account.merge(conflict_keys, on=["CNPJ_CIA", "ANO"], how="left")
                account = account.loc[account["_conflict"].isna()].drop(columns=["_conflict"])
                diagnostics["conflicts"].append(conflict_rows)

            extracted = account[["CNPJ_CIA", "ANO", component_col]].drop_duplicates()
            if extracted.duplicated(KEY_COLS).any():
                duplicated = extracted.loc[extracted.duplicated(KEY_COLS, keep=False)].sort_values(KEY_COLS)
                raise ValueError(
                    f"Duplicidade no componente {component_col} depois dos filtros:\n"
                    + duplicated.to_string(index=False)
                )
            all_components.append(extracted)
            if not diff_desc.empty:
                diagnostics["different_descriptions"].append(diff_desc)

    result = panel[KEY_COLS].copy()
    for component_col, _code, _description in components:
        frames = [df for df in all_components if component_col in df.columns]
        component_values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEY_COLS + [component_col])
        result = result.merge(component_values, on=KEY_COLS, how="left")

    diagnostics["conflicts"] = pd.concat(diagnostics["conflicts"], ignore_index=True) if diagnostics["conflicts"] else pd.DataFrame()
    diagnostics["different_descriptions"] = (
        pd.concat(diagnostics["different_descriptions"], ignore_index=True)
        if diagnostics["different_descriptions"]
        else pd.DataFrame()
    )
    return result, diagnostics


def compute_x15(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    components, diagnostics = extract_component_accounts(
        panel,
        "BPP",
        [
            ("reservas_de_lucros", "2.03.04", DESC["reservas_lucros"]),
            ("lucros_prejuizos_acumulados", "2.03.05", DESC["lucros_acumulados"]),
        ],
    )
    components["X15"] = np.nan
    has_both = components["reservas_de_lucros"].notna() & components["lucros_prejuizos_acumulados"].notna()
    components.loc[has_both, "X15"] = (
        components.loc[has_both, "reservas_de_lucros"]
        + components.loc[has_both, "lucros_prejuizos_acumulados"]
    )
    components["componente_ausente"] = ""
    components.loc[components["reservas_de_lucros"].isna(), "componente_ausente"] += "2.03.04;"
    components.loc[components["lucros_prejuizos_acumulados"].isna(), "componente_ausente"] += "2.03.05;"
    return components[KEY_COLS + ["X15"]], components, diagnostics


def compute_x17(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    components, diagnostics = extract_component_accounts(
        panel,
        "BPP",
        [
            ("passivo_circulante", "2.01", DESC["current_liabilities"]),
            ("passivo_nao_circulante", "2.02", DESC["passivo_nao_circulante"]),
        ],
    )
    components["X17"] = np.nan
    has_both = components["passivo_circulante"].notna() & components["passivo_nao_circulante"].notna()
    components.loc[has_both, "X17"] = (
        components.loc[has_both, "passivo_circulante"]
        + components.loc[has_both, "passivo_nao_circulante"]
    )
    components["componente_ausente"] = ""
    components.loc[components["passivo_circulante"].isna(), "componente_ausente"] += "2.01;"
    components.loc[components["passivo_nao_circulante"].isna(), "componente_ausente"] += "2.02;"
    return components[KEY_COLS + ["X17"]], components, diagnostics


def compute_diagnostics_x13(panel: pd.DataFrame) -> pd.DataFrame:
    values, _diagnostics = extract_account(
        panel,
        {"var": "resultado_bruto_3_03", "statement": "DRE", "code": "3.03", "desc": DESC["resultado_bruto"], "abs": False},
    )
    dre_302, _ = extract_account(
        panel,
        {"var": "cogs_original_3_02", "statement": "DRE", "code": "3.02", "desc": DESC["cogs"], "abs": False},
    )
    compare = panel[KEY_COLS + ["DENOM_CIA", "X9", "X2", "X13"]].merge(values, on=KEY_COLS, how="left")
    compare = compare.merge(dre_302, on=KEY_COLS, how="left")
    compare["x9_minus_x2_vs_303"] = compare["X13"] - compare["resultado_bruto_3_03"]
    compare["identity_301_plus_302"] = compare["X9"] + compare["cogs_original_3_02"]
    compare["identity_diff"] = compare["identity_301_plus_302"] - compare["resultado_bruto_3_03"]
    compare["diferenca_absoluta"] = compare["x9_minus_x2_vs_303"].abs()
    compare["diferenca_percentual"] = np.where(
        compare["resultado_bruto_3_03"].ne(0),
        compare["diferenca_absoluta"] / compare["resultado_bruto_3_03"].abs() * 100,
        np.nan,
    )
    return compare


def compute_diagnostics_x10(panel: pd.DataFrame) -> pd.DataFrame:
    components, _diagnostics = extract_component_accounts(
        panel,
        "BPA",
        [
            ("ativo_circulante", "1.01", DESC["ativo_circulante"]),
            ("ativo_nao_circulante", "1.02", "Ativo N\u00e3o Circulante"),
        ],
    )
    diagnostic = panel[KEY_COLS + ["DENOM_CIA", "X10"]].merge(components, on=KEY_COLS, how="left")
    diagnostic["soma_1_01_1_02"] = diagnostic["ativo_circulante"] + diagnostic["ativo_nao_circulante"]
    diagnostic["diferenca"] = diagnostic["X10"] - diagnostic["soma_1_01_1_02"]
    return diagnostic


def compute_diagnostics_x17(panel: pd.DataFrame) -> pd.DataFrame:
    pl_values, _diagnostics = extract_account(
        panel,
        {"var": "patrimonio_liquido_2_03", "statement": "BPP", "code": "2.03", "desc": DESC["patrimonio_liquido"], "abs": False},
    )
    diagnostic = panel[KEY_COLS + ["DENOM_CIA", "X10", "X17"]].merge(pl_values, on=KEY_COLS, how="left")
    diagnostic["ativo_total_menos_pl"] = diagnostic["X10"] - diagnostic["patrimonio_liquido_2_03"]
    diagnostic["diferenca"] = diagnostic["ativo_total_menos_pl"] - diagnostic["X17"]
    return diagnostic


def analyze_capital_x8(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = capital_path(year)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
        capital = read_capital(path)
        year_keys = panel.loc[panel["ANO"].eq(year), ["CNPJ_CIA", "ANO", "DT_REFER"]].copy()
        capital = capital.merge(year_keys, on=["CNPJ_CIA", "DT_REFER"], how="inner")
        capital["VERSAO"] = pd.to_numeric(capital["VERSAO"], errors="coerce")
        latest = capital.groupby(["CNPJ_CIA", "ANO", "DT_REFER"], dropna=False)["VERSAO"].transform("max")
        capital = capital.loc[capital["VERSAO"].eq(latest)].copy()
        frames.append(capital)
    return pd.concat(frames, ignore_index=True)


def validate_panel(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    if len(before) != len(after):
        raise ValueError(f"Quantidade de linhas mudou: antes={len(before)}, depois={len(after)}")
    if after.duplicated(KEY_COLS).any():
        raise ValueError("Existem duplicidades CNPJ_CIA + ANO.")

    expected_years = set(YEARS)
    years_ok = after.groupby("CNPJ_CIA")["ANO"].apply(lambda values: set(values) == expected_years)
    counts = after.groupby("CNPJ_CIA").size()
    if not years_ok.all() or not counts.eq(5).all():
        raise ValueError("Painel deixou de ser balanceado.")

    protected_cols = ["CNPJ_CIA", "CD_CVM", "DENOM_CIA", "ANO", "DT_REFER"]
    before_protected = before[protected_cols].sort_values(KEY_COLS).reset_index(drop=True)
    after_protected = after[protected_cols].sort_values(KEY_COLS).reset_index(drop=True)
    if not before_protected.equals(after_protected):
        raise ValueError("Colunas de identificacao foram alteradas indevidamente.")

    return {
        "linhas_antes": len(before),
        "linhas_depois": len(after),
        "empresas": int(after["CNPJ_CIA"].nunique()),
        "cnpjs_exatamente_5": int(counts.eq(5).sum()),
        "anos_corretos_por_cnpj": bool(years_ok.all()),
        "duplicidades_cnpj_ano": int(after.duplicated(KEY_COLS).sum()),
    }


def save_and_reopen(panel: pd.DataFrame) -> pd.DataFrame:
    panel[PANEL_COLS].to_csv(PANEL_PATH, sep=";", encoding="utf-8-sig", index=False)
    reopened = pd.read_csv(PANEL_PATH, sep=";", encoding="utf-8-sig", dtype={"CNPJ_CIA": "string", "CD_CVM": "string"})
    if reopened.shape[1] == 1:
        raise ValueError("Erro de delimitador: CSV reaberto com apenas uma coluna.")
    return reopened


def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in X_COLS:
        non_null = panel[variable].dropna()
        rows.append(
            {
                "VARIAVEL": variable,
                "PREENCHIDO": int(panel[variable].notna().sum()),
                "NaN": int(panel[variable].isna().sum()),
                "COBERTURA": f"{panel[variable].notna().mean() * 100:.2f}%",
                "MIN": non_null.min() if not non_null.empty else np.nan,
                "MEDIANA": non_null.median() if not non_null.empty else np.nan,
                "MAX": non_null.max() if not non_null.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def coverage_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in X_COLS:
        for year in YEARS:
            year_df = panel.loc[panel["ANO"].eq(year)]
            rows.append(
                {
                    "VARIAVEL": variable,
                    "ANO": year,
                    "PREENCHIDO": int(year_df[variable].notna().sum()),
                    "NaN": int(year_df[variable].isna().sum()),
                    "COBERTURA": f"{year_df[variable].notna().mean() * 100:.2f}%",
                }
            )
    return pd.DataFrame(rows)


def print_report(
    panel: pd.DataFrame,
    reopened: pd.DataFrame,
    integrity: dict[str, object],
    account_diagnostics: list[dict[str, object]],
    x15_components: pd.DataFrame,
    x17_components: pd.DataFrame,
    x13_diag: pd.DataFrame,
    x10_diag: pd.DataFrame,
    x17_diag: pd.DataFrame,
    capital_x8: pd.DataFrame,
) -> None:
    print("RELATORIO - VARIAVEIS DO PAINEL 2021-2025")
    print()
    print(f"arquivo_salvo: {PANEL_PATH}")
    print("salvamento: sep=';', encoding='utf-8-sig', index=False")
    print()
    print("CSV reaberto com sep=';'")
    print(f"linhas: {len(reopened)}")
    print(f"colunas: {reopened.shape[1]}")
    print(f"nomes_colunas: {reopened.columns.tolist()}")
    print()
    print("Integridade do painel")
    print(f"linhas_antes: {integrity['linhas_antes']}")
    print(f"linhas_depois: {integrity['linhas_depois']}")
    print(f"empresas: {integrity['empresas']}")
    print(f"CNPJs com exatamente 5 observacoes: {integrity['cnpjs_exatamente_5']}")
    print(f"CNPJs com anos 2021-2025: {integrity['anos_corretos_por_cnpj']}")
    print(f"duplicidades CNPJ_CIA + ANO: {integrity['duplicidades_cnpj_ano']}")
    print("colunas de identificacao preservadas: OK")
    print()
    print("Cobertura geral por variavel")
    print(coverage_table(panel).to_string(index=False))
    print()
    print("Cobertura por ano")
    print(coverage_by_year(panel).to_string(index=False))
    print()
    print("Diagnosticos por conta extraida")
    for diag in account_diagnostics:
        variable = diag["variable"]
        diff_desc = diag["different_descriptions"]
        conflicts = diag["conflicts"]
        unknown_scales = {year: scales for year, scales in diag["unknown_scales"].items() if scales}
        conflict_count = conflicts[KEY_COLS].drop_duplicates().shape[0] if not conflicts.empty else 0
        diff_desc_count = diff_desc[KEY_COLS].drop_duplicates().shape[0] if not diff_desc.empty else 0
        print(
            f"{variable}: duplicidades_exatas={diag['exact_duplicates']}; "
            f"conflitos={conflict_count}; escalas_desconhecidas={unknown_scales if unknown_scales else 0}; "
            f"codigo_com_outra_descricao={diff_desc_count}"
        )
        if variable in {"X2", "X3", "X6", "X12"}:
            print(f"{variable}: sinais_originais={diag['original_signs']}")
    print()
    print("X8 - market_value")
    print("X8 mantido como NaN: composicao_capital informa quantidade de acoes, mas nao contem preco de mercado.")
    print(f"observacoes com composicao_capital disponivel no painel: {capital_x8[KEY_COLS].drop_duplicates().shape[0]}")
    print()
    print("Validacoes derivadas")
    x13_comp = x13_diag.dropna(subset=["X13", "resultado_bruto_3_03"])
    x13_equal = int(np.isclose(x13_comp["x9_minus_x2_vs_303"], 0, atol=1e-6).sum()) if not x13_comp.empty else 0
    x13_diff = int((~np.isclose(x13_comp["x9_minus_x2_vs_303"], 0, atol=1e-6)).sum()) if not x13_comp.empty else 0
    identity_equal = int(np.isclose(x13_comp["identity_diff"], 0, atol=1e-6).sum()) if not x13_comp.empty else 0
    identity_diff = int((~np.isclose(x13_comp["identity_diff"], 0, atol=1e-6)).sum()) if not x13_comp.empty else 0
    print(f"X13 = X9 - X2 comparado com DRE 3.03: iguais={x13_equal}, diferentes={x13_diff}")
    print(f"Identidade DRE 3.03 ~= 3.01 + 3.02 original: iguais={identity_equal}, diferentes={identity_diff}")
    x10_comp = x10_diag.dropna(subset=["X10", "soma_1_01_1_02"])
    x10_equal = int(np.isclose(x10_comp["diferenca"], 0, atol=1e-6).sum()) if not x10_comp.empty else 0
    x10_diff = int((~np.isclose(x10_comp["diferenca"], 0, atol=1e-6)).sum()) if not x10_comp.empty else 0
    print(f"X10 vs 1.01 + 1.02: iguais={x10_equal}, diferentes={x10_diff}")
    x17_comp = x17_diag.dropna(subset=["X10", "X17", "patrimonio_liquido_2_03"])
    x17_equal = int(np.isclose(x17_comp["diferenca"], 0, atol=1e-6).sum()) if not x17_comp.empty else 0
    x17_diff_count = int((~np.isclose(x17_comp["diferenca"], 0, atol=1e-6)).sum()) if not x17_comp.empty else 0
    print(f"X17 vs X10 - Patrimonio Liquido: iguais={x17_equal}, diferentes={x17_diff_count}")
    print()
    print("Componentes X15 - exemplos")
    print(
        x15_components.merge(panel[KEY_COLS + ["DENOM_CIA"]], on=KEY_COLS, how="left")[
            ["CNPJ_CIA", "DENOM_CIA", "ANO", "reservas_de_lucros", "lucros_prejuizos_acumulados", "X15"]
        ].head(10).to_string(index=False)
    )
    print()
    print("Componentes X17 - exemplos")
    print(
        x17_components.merge(panel[KEY_COLS + ["DENOM_CIA"]], on=KEY_COLS, how="left")[
            ["CNPJ_CIA", "DENOM_CIA", "ANO", "passivo_circulante", "passivo_nao_circulante", "X17"]
        ].head(10).to_string(index=False)
    )
    print()
    print("Amostra final - primeiras 10 linhas")
    print(panel.head(10).to_string(index=False))
    print()
    print("Amostra final - ultimas 10 linhas")
    print(panel.tail(10).to_string(index=False))


def run_panel_variables() -> None:
    panel = read_panel()
    before = panel.copy(deep=True)
    account_diagnostics = []

    for spec in ACCOUNT_SPECS:
        values, diagnostics = extract_account(panel, spec)
        panel = assign_variable(panel, values, str(spec["var"]))
        account_diagnostics.append(diagnostics)

    x15_values, x15_components, x15_diag = compute_x15(panel)
    panel = assign_variable(panel, x15_values, "X15")
    x15_diag["variable"] = "X15"
    account_diagnostics.append(x15_diag)

    x17_values, x17_components, x17_diag_components = compute_x17(panel)
    panel = assign_variable(panel, x17_values, "X17")
    x17_diag_components["variable"] = "X17"
    account_diagnostics.append(x17_diag_components)

    panel["X4"] = np.where(panel["X12"].notna() & panel["X3"].notna(), panel["X12"] + panel["X3"], np.nan)
    panel["X13"] = np.where(panel["X9"].notna() & panel["X2"].notna(), panel["X9"] - panel["X2"], np.nan)
    panel["X16"] = pd.to_numeric(panel["X9"], errors="coerce")
    panel["X18"] = np.where(panel["X9"].notna() & panel["X4"].notna(), panel["X9"] - panel["X4"], np.nan)
    panel["X8"] = np.nan

    x13_diag = compute_diagnostics_x13(panel)
    x10_diag = compute_diagnostics_x10(panel)
    x17_diag = compute_diagnostics_x17(panel)
    capital_x8 = analyze_capital_x8(panel)

    panel = panel[PANEL_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    integrity = validate_panel(before, panel)
    reopened = save_and_reopen(panel)
    print_report(
        panel,
        reopened,
        integrity,
        account_diagnostics,
        x15_components,
        x17_components,
        x13_diag,
        x10_diag,
        x17_diag,
        capital_x8,
    )


if __name__ == "__main__":
    run_panel_variables()
