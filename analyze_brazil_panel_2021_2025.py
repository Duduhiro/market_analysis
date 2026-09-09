from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BRAZIL_DIR = BASE_DIR / "brazil_database"
OUTPUT_PATH = BASE_DIR / "empresas_painel_2021_2025.csv"

YEARS = [2021, 2022, 2023, 2024, 2025]
STATEMENTS = ["BPA", "BPP", "DRE", "DVA"]


def read_metadata(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype={"CNPJ_CIA": "string", "CD_CVM": "string"},
        usecols=lambda col: col in {"CNPJ_CIA", "CD_CVM", "DENOM_CIA"},
    )


def statement_path(year: int, statement: str) -> Path:
    return BRAZIL_DIR / str(year) / f"dfp_cia_aberta_{statement}_con_{year}.csv"


def capital_path(year: int) -> Path:
    return BRAZIL_DIR / str(year) / f"dfp_cia_aberta_composicao_capital_{year}.csv"


def cnpj_set(df: pd.DataFrame) -> set[str]:
    return set(df["CNPJ_CIA"].dropna().astype("string").str.strip())


def analyze_years() -> tuple[dict[int, dict[str, set[str]]], pd.DataFrame, dict[int, set[str]]]:
    yearly_statement_sets: dict[int, dict[str, set[str]]] = {}
    valid_by_year: dict[int, set[str]] = {}
    rows = []

    for year in YEARS:
        statement_sets = {}
        for statement in STATEMENTS:
            path = statement_path(year, statement)
            if not path.exists():
                raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

            df = read_metadata(path)
            statement_sets[statement] = cnpj_set(df)

        valid = set.intersection(*(statement_sets[statement] for statement in STATEMENTS))
        yearly_statement_sets[year] = statement_sets
        valid_by_year[year] = valid
        rows.append(
            {
                "ANO": year,
                "BPA": len(statement_sets["BPA"]),
                "BPP": len(statement_sets["BPP"]),
                "DRE": len(statement_sets["DRE"]),
                "DVA": len(statement_sets["DVA"]),
                "INTERSECAO_BPA_BPP_DRE_DVA": len(valid),
                "ELIMINADAS_NO_ANO_POR_NAO_ESTAR_NAS_4_DEMONSTRACOES": len(
                    set.union(*(statement_sets[statement] for statement in STATEMENTS)) - valid
                ),
            }
        )

    return yearly_statement_sets, pd.DataFrame(rows), valid_by_year


def analyze_presence_by_year(valid_by_year: dict[int, set[str]]) -> tuple[set[str], pd.DataFrame, dict[int, int]]:
    all_valid_companies = set.union(*(valid_by_year[year] for year in YEARS))
    presence_rows = []
    for cnpj in sorted(all_valid_companies):
        years_present = [year for year in YEARS if cnpj in valid_by_year[year]]
        presence_rows.append(
            {
                "CNPJ_CIA": cnpj,
                "QTD_ANOS_PRESENTE": len(years_present),
                "ANOS_PRESENTE": ",".join(str(year) for year in years_present),
            }
        )

    presence = pd.DataFrame(presence_rows)
    counts = {
        years_count: int((presence["QTD_ANOS_PRESENTE"] == years_count).sum())
        for years_count in range(1, 6)
    }
    balanced = set(presence.loc[presence["QTD_ANOS_PRESENTE"].eq(5), "CNPJ_CIA"])
    return balanced, presence, counts


def metadata_for_balanced_companies(balanced: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_frames = []
    for year in YEARS:
        for statement in STATEMENTS:
            df = read_metadata(statement_path(year, statement))
            df = df.loc[df["CNPJ_CIA"].isin(balanced), ["CNPJ_CIA", "CD_CVM", "DENOM_CIA"]].drop_duplicates()
            df["ANO"] = year
            df["DEMONSTRACAO"] = statement
            metadata_frames.append(df)

    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata["CNPJ_CIA"] = metadata["CNPJ_CIA"].astype("string").str.strip()
    metadata["CD_CVM"] = metadata["CD_CVM"].astype("string").str.strip()

    cd_cvm_counts = metadata.groupby("CNPJ_CIA")["CD_CVM"].nunique(dropna=True).reset_index(name="QTD_CD_CVM")
    cd_cvm_conflicts = cd_cvm_counts.loc[cd_cvm_counts["QTD_CD_CVM"].gt(1), ["CNPJ_CIA"]].merge(
        metadata[["CNPJ_CIA", "CD_CVM", "DENOM_CIA", "ANO", "DEMONSTRACAO"]].drop_duplicates(),
        on="CNPJ_CIA",
        how="left",
    )

    names_2025 = (
        metadata.loc[metadata["ANO"].eq(2025), ["CNPJ_CIA", "DENOM_CIA"]]
        .dropna(subset=["DENOM_CIA"])
        .drop_duplicates("CNPJ_CIA", keep="last")
    )
    cd_cvm_latest = (
        metadata.sort_values(["ANO", "DEMONSTRACAO"])
        .dropna(subset=["CD_CVM"])
        .drop_duplicates("CNPJ_CIA", keep="last")[["CNPJ_CIA", "CD_CVM"]]
    )

    final = pd.DataFrame({"CNPJ_CIA": sorted(balanced)})
    final = final.merge(cd_cvm_latest, on="CNPJ_CIA", how="left")
    final = final.merge(names_2025, on="CNPJ_CIA", how="left")

    missing_name = final["DENOM_CIA"].isna()
    if missing_name.any():
        fallback_names = (
            metadata.sort_values(["ANO", "DEMONSTRACAO"])
            .dropna(subset=["DENOM_CIA"])
            .drop_duplicates("CNPJ_CIA", keep="last")[["CNPJ_CIA", "DENOM_CIA"]]
        )
        final = final.drop(columns=["DENOM_CIA"]).merge(fallback_names, on="CNPJ_CIA", how="left")

    name_variations = (
        metadata.groupby("CNPJ_CIA")["DENOM_CIA"]
        .nunique(dropna=True)
        .reset_index(name="QTD_DENOM_CIA")
    )
    name_variations = name_variations.loc[name_variations["QTD_DENOM_CIA"].gt(1), ["CNPJ_CIA"]].merge(
        metadata[["CNPJ_CIA", "DENOM_CIA", "ANO", "DEMONSTRACAO"]].drop_duplicates(),
        on="CNPJ_CIA",
        how="left",
    )

    return final[["CNPJ_CIA", "CD_CVM", "DENOM_CIA"]], cd_cvm_conflicts, name_variations


def analyze_capital_composition(balanced: set[str]) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        path = capital_path(year)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

        df = pd.read_csv(path, sep=";", encoding="latin1", dtype={"CNPJ_CIA": "string"}, usecols=["CNPJ_CIA"])
        available = cnpj_set(df)
        for cnpj in sorted(balanced):
            rows.append({"CNPJ_CIA": cnpj, "ANO": year, "TEM_COMPOSICAO_CAPITAL": cnpj in available})

    availability = pd.DataFrame(rows)
    summary = (
        availability.groupby("CNPJ_CIA")["TEM_COMPOSICAO_CAPITAL"]
        .sum()
        .astype(int)
        .reset_index(name="ANOS_COM_COMPOSICAO_CAPITAL")
    )
    summary["FAIXA"] = pd.cut(
        summary["ANOS_COM_COMPOSICAO_CAPITAL"],
        bins=[-1, 3, 4, 5],
        labels=["3 anos ou menos", "4 anos", "5 anos"],
    )
    return summary


def validate_final(final: pd.DataFrame, balanced: set[str], yearly_statement_sets: dict[int, dict[str, set[str]]]) -> None:
    duplicates = int(final["CNPJ_CIA"].duplicated().sum())
    if duplicates:
        raise ValueError(f"Arquivo final teria {duplicates} CNPJ_CIA duplicados.")

    missing = []
    for cnpj in final["CNPJ_CIA"]:
        for year in YEARS:
            for statement in STATEMENTS:
                if cnpj not in yearly_statement_sets[year][statement]:
                    missing.append({"CNPJ_CIA": cnpj, "ANO": year, "DEMONSTRACAO": statement})

    if missing:
        raise ValueError("Validacao falhou: CNPJs ausentes em demonstracoes obrigatorias:\n" + pd.DataFrame(missing).to_string(index=False))

    if set(final["CNPJ_CIA"]) != balanced:
        raise ValueError("Validacao falhou: arquivo final difere do conjunto balanceado calculado.")


def print_report(
    yearly_counts: pd.DataFrame,
    valid_by_year: dict[int, set[str]],
    balanced: set[str],
    presence: pd.DataFrame,
    presence_counts: dict[int, int],
    capital_summary: pd.DataFrame,
    final: pd.DataFrame,
    cd_cvm_conflicts: pd.DataFrame,
    name_variations: pd.DataFrame,
) -> None:
    print("RELATORIO - UNIVERSO CVM 2021-2025")
    print()
    print("ANO | BPA | BPP | DRE | DVA | INTERSECAO_BPA_BPP_DRE_DVA")
    print(
        yearly_counts[
            ["ANO", "BPA", "BPP", "DRE", "DVA", "INTERSECAO_BPA_BPP_DRE_DVA"]
        ].to_string(index=False)
    )
    print()
    print(f"Empresas presentes simultaneamente em 2021-2025: {len(balanced)}")
    print()
    print(f"empresas presentes em 5 anos: {presence_counts[5]}")
    print(f"empresas presentes em exatamente 4 anos: {presence_counts[4]}")
    print(f"empresas presentes em exatamente 3 anos: {presence_counts[3]}")
    print(f"empresas presentes em exatamente 2 anos: {presence_counts[2]}")
    print(f"empresas presentes em apenas 1 ano: {presence_counts[1]}")
    print()
    print("empresas_eliminadas_em_cada_ano_por_nao_estarem_nas_4_demonstracoes:")
    print(
        yearly_counts[
            ["ANO", "ELIMINADAS_NO_ANO_POR_NAO_ESTAR_NAS_4_DEMONSTRACOES"]
        ].to_string(index=False)
    )
    print()
    print("empresas_eliminadas_na_intersecao_entre_anos:")
    print(f"universo_uniao_empresas_validas_anuais: {len(presence)}")
    print(f"eliminadas_por_nao_estarem_nos_5_anos: {int((presence['QTD_ANOS_PRESENTE'] < 5).sum())}")
    for year in YEARS:
        print(f"validas_{year}: {len(valid_by_year[year])}")
    print()
    print("composicao_de_capital_no_painel_balanceado:")
    capital_counts = capital_summary["ANOS_COM_COMPOSICAO_CAPITAL"].value_counts().to_dict()
    print(f"nos_5_anos: {capital_counts.get(5, 0)}")
    print(f"em_4_anos: {capital_counts.get(4, 0)}")
    print(f"em_3_anos_ou_menos: {int(capital_summary['ANOS_COM_COMPOSICAO_CAPITAL'].le(3).sum())}")
    print()
    print(f"arquivo_controle_gerado: {OUTPUT_PATH}")
    print(f"cnpj_duplicados_no_arquivo_final: {int(final['CNPJ_CIA'].duplicated().sum())}")
    print(f"validacao_presenca_em_todas_as_demonstracoes_2021_2025: OK")
    print()
    print(f"casos_com_cd_cvm_variando_para_o_mesmo_cnpj: {cd_cvm_conflicts['CNPJ_CIA'].nunique() if not cd_cvm_conflicts.empty else 0}")
    if not cd_cvm_conflicts.empty:
        print("detalhe_cd_cvm_variando:")
        print(cd_cvm_conflicts.to_string(index=False))
    print()
    print(f"casos_com_denom_cia_variando_entre_anos: {name_variations['CNPJ_CIA'].nunique() if not name_variations.empty else 0}")
    if not name_variations.empty:
        print("exemplos_denom_cia_variando:")
        print(name_variations.head(40).to_string(index=False))
    print()
    print("20 exemplos das empresas que permaneceram:")
    print(final.head(20).to_string(index=False))
    print()
    eliminated = presence.loc[presence["QTD_ANOS_PRESENTE"].lt(5)].sort_values(["QTD_ANOS_PRESENTE", "CNPJ_CIA"])
    print("20 exemplos de empresas eliminadas por nao estarem presentes nos cinco anos:")
    print(eliminated.head(20).to_string(index=False))


def main() -> None:
    yearly_statement_sets, yearly_counts, valid_by_year = analyze_years()
    balanced, presence, presence_counts = analyze_presence_by_year(valid_by_year)
    final, cd_cvm_conflicts, name_variations = metadata_for_balanced_companies(balanced)
    capital_summary = analyze_capital_composition(balanced)
    validate_final(final, balanced, yearly_statement_sets)

    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print_report(
        yearly_counts,
        valid_by_year,
        balanced,
        presence,
        presence_counts,
        capital_summary,
        final,
        cd_cvm_conflicts,
        name_variations,
    )


if __name__ == "__main__":
    main()
