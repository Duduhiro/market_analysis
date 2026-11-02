# Relatório de Progresso Metodológico e Resultados Preliminares

## 1. Engenharia de Atributos e Pré-processamento

* **Normalização das Variáveis:** A base norte-americana bruta (*US Company Bankruptcy Prediction*) continha variáveis nominais em dólares ($X_1$ a $X_{18}$). Para viabilizar a comparação com o mercado brasileiro e eliminar distorções de escala monetária, os dados foram transformados em **indicadores financeiros adimensionais** baseados na literatura clássica (Altman, Ohlson, Beaver):


* *Liquidez:* Liquidez Corrente, Liquidez Seca e Capital de Giro sobre Ativo Total.

* *Rentabilidade:* ROA, ROE, Margem Bruta, Margem Líquida e Margem Operacional.

* *Endividamento/Estrutura:* Dívida Total/Ativo, Dívida de Longo Prazo/Ativo e Valor de Mercado sobre Passivo Total.

* *Eficiência/Atividade:* Giro do Ativo e razões de custos operacionais sobre faturamento.

* **Tratamento Numérico:** Realizou-se tratamento de infinitos ($\pm\infty$) decorrentes de denominadores nulos, imputação pela mediana, truncamento de valores extremos (*Winsorization* nos percentis 1% e 99%) e padronização com `RobustScaler`.

## 2. Modelagem e Estratégia Experimental

* **Divisão Estratificada (*Hold-out* 70/30):** Partição preservando a taxa natural de eventos de falência (~$6{,}63\%$) no conjunto de teste ($23.605$ instâncias não vistas).
* **Tratamento de Desbalanceamento:** Aplicação da técnica **SMOTE** (*Synthetic Minority Over-sampling Technique*) restrita ao conjunto de treinamento para evitar vazamento de dados (*data leakage*).

* **Modelos Desenvolvidos:**
1. *Regressão Logística (Logit):* Modelo paramétrico tradicional de linha de base (*baseline*).

2. *Random Forest:* Ensemble por agregação (*bagging*) com ajuste de pesos de classes.

3. *XGBoost:* Algoritmo de *gradient boosting* otimizado com ponderação de penalidades para a classe minoritária e grade de hiperparâmetros regularizadores (`gamma`, `reg_lambda`, `reg_alpha`, `min_child_weight`).

## 3. Avaliação de Desempenho e Resultados do Benchmark

A avaliação no conjunto de teste revelou clara superioridade das arquiteturas em ensemble sobre a abordagem linear:

* **XGBoost (Modelo Campeão):** Atingiu a maior capacidade discriminativa global (**AUC-ROC = 0,7846**) e o maior **Recall de Falência (60,41%)**, capturando $946$ das $1.566$ falências reais no teste. Na gestão de risco de crédito, a maximização do Recall é o critério mais relevante, pois minimiza o Erro Tipo II (não identificar a insolvência).
* **Random Forest:** Apresentou acurácia geral elevada ($87,56\%$) e maior precisão ($21,85\%$), porém demonstrou sensibilidade inferior (**Recall = 33,97%**), deixando passar $1.034$ casos de falência.
* **Regressão Logística:** Teve o desempenho mais fraco (**AUC-ROC = 0,6278**), comprovando a ineficácia de funções lineares rígidas para separar relações complexas e não lineares nos balanços.

---

## 4. Estudo Comparativo: Dados Tratados (Rácios) vs. Dados Brutos (*Untreated*)

Para verificar se a criação dos parâmetros financeiros realmente agregou valor, foi conduzido um experimento paralelo treinando os mesmos algoritmos diretamente nas variáveis brutas ($X_1$ a $X_{18}$):

* **O Falso Ganho nos Dados Brutos:** Embora os dados brutos tenham apresentado AUC marginalmente superior no XGBoost ($0{,}7997$ vs. $0{,}7846$), o gráfico de explicabilidade **SHAP** comprovou que isso se deu por um atalho estatístico (*shortcut learning*): o modelo dependia exclusivamente do `market_value` absoluto em dólares, ignorando indicadores contábeis essenciais.
* **Colapso da Regressão Logística Bruta:** Devido à severa multicolinearidade entre variáveis nominais de balanço (correlações acima de $0{,}90$), o modelo bruto gerou **$14.407$ falsos alarmes** (classificou $65\%$ da base como falida). A criação dos rácios reduziu os falsos positivos para $8.687$ (uma eliminação de **$5.720$ alarmes falsos**), estabilizando o estimador estatístico.
* **Invariância de Escala:** A criação dos parâmetros adimensionais provou ser indispensável: sem eles, seria matematicamente impossível transferir o modelo norte-americano para empresas brasileiras cotadas em Reais (R$) e com portes de capital substancialmente menores.

## 5. Explicabilidade (XAI via SHAP e *Feature Importance*)

A análise do XGBoost via valores SHAP forneceu respaldo empírico alinhado à teoria contábil:

1. **Solvência de Mercado (`market_val_to_liabilities`):** Foi a variável mais determinante; baixos valores dessa razão disparam a probabilidade marginal de falência.
2. **Capacidade Operacional (`EBIT_to_assets` e `gross_margin`):** Apontados como gatilhos primários de deterioração: empresas com margens comprimidas e retornos operacionais negativos deslocam-se consistentemente para a faixa de insolvência.
3. **Alavancagem de Longo Prazo (`long_term_debt_to_assets`):** Relação diretamente proporcional com o risco de crédito, confirmando a teoria clássica de insolvência por excesso de endividamento oneroso.

## 6. Próximo Passo

Com os modelos calibrados, validados e serializados em disco, o pipeline entra em sua fase final: **Módulo 4 (Transferibilidade *Cross-Market*)**, no qual os demonstrativos contábeis das companhias brasileiras (B3/CVM) serão ingeridos para geração do *score* probabilístico de risco e classificação de insolvência sem a necessidade de rótulo local prévio.