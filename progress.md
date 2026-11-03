# Relatório de Progresso Metodológico e Resultados

## 1. Objetivo e preparação dos dados

O projeto avalia modelos de previsão de falência treinados com empresas dos Estados Unidos e sua capacidade de generalização para empresas brasileiras. As variáveis contábeis nominais são convertidas em indicadores financeiros relativos de liquidez, rentabilidade, endividamento e atividade.

O tratamento de dados agora preserva valores ausentes durante a engenharia de atributos, sinaliza linhas sem indicadores financeiros utilizáveis e deixa imputação, truncamento e escala para depois da divisão de treino e teste. Assim, os parâmetros de pré-processamento são ajustados apenas no conjunto de treino.

Há uma diferença relevante entre as bases: `X8` (`market_value`) está totalmente ausente na base brasileira. Portanto, `market_val_to_liabilities` não contém informação brasileira e, no experimento atual, é imputado com a mediana aprendida nos dados americanos. Esse resultado deve ser interpretado com cautela; a imputação não recupera o valor de mercado brasileiro.

## 2. Benchmark interno nos Estados Unidos

As métricas abaixo são do holdout americano registrado nos resultados atuais. Elas medem desempenho dentro do mercado de origem, não transferibilidade para o Brasil.

| Dados | Modelo | Acurácia | Precisão | Recall | AUC-ROC |
|---|---|---:|---:|---:|---:|
| Tratados | Regressão Logística | 0,6056 | 0,0950 | 0,5798 | 0,6279 |
| Tratados | Random Forest | 0,8742 | 0,2160 | 0,3410 | 0,7758 |
| Tratados | XGBoost | 0,7412 | 0,1545 | 0,6488 | 0,7714 |
| Brutos | Regressão Logística | 0,4026 | 0,0846 | 0,8148 | 0,6772 |
| Brutos | Random Forest | 0,8771 | 0,2434 | 0,4042 | 0,7967 |
| Brutos | XGBoost | 0,7411 | 0,1571 | 0,6648 | 0,7839 |

Os dados brutos tiveram AUC e recall maiores nos três modelos neste holdout; o Random Forest bruto obteve a maior AUC (0,7967), e a Regressão Logística bruta, o maior recall (0,8148), mas com muitos falsos positivos. A comparação usa as mesmas 23.605 observações de teste e os mesmos rótulos para ambas as representações. A escolha do modelo depende do custo relativo de falsos negativos e falsos positivos; acurácia isolada é insuficiente diante do desbalanceamento.

## 3. Experimento cross-market: Estados Unidos → Brasil

O script `cross_market.py` separa 30% da base americana estratificada e treina com os 70% restantes. O pré-processamento é ajustado no treino americano e aplicado à base brasileira; o SMOTE é restrito ao treino americano. Portanto, o treinamento é exclusivamente americano, mas não usa 100% das linhas americanas.

A base brasileira contém 1.850 observações, das quais 180 estão rotuladas como falidas. O modelo não usa esses rótulos para treinamento, mas o script os usa para calcular métricas de validação externa. Assim, o experimento é uma avaliação cross-market rotulada, não apenas uma geração cega de scores.

| Modelo | Acurácia | Precisão | Recall | AUC-ROC | Falências detectadas | Alarmes falsos |
|---|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,5670 | 0,1045 | 0,4556 | 0,4984 | 82/180 | 703 |
| Random Forest | 0,9049 | 0,5667 | 0,0944 | 0,7260 | 17/180 | 13 |
| XGBoost | 0,8173 | 0,2238 | 0,3556 | 0,7027 | 64/180 | 222 |

O Random Forest obteve a maior AUC brasileira (0,7260) e poucos alarmes falsos, mas detectou apenas 17 das 180 falências no limiar atual de 0,5. O XGBoost detectou 64 falências, com 222 alarmes falsos. A Regressão Logística teve AUC de 0,4984, próxima ao acaso, apesar de identificar mais casos nesse limiar e gerar muitos falsos positivos. A acurácia de 0,9049 do Random Forest, portanto, não representa boa detecção da classe minoritária.

Os resultados indicam transferibilidade parcial e dependente do modelo e do limiar, não uma transferência robusta já demonstrada. Os artefatos estão em `results_cross_market/` e os modelos cross-market em `models/`.

## 4. Próximos passos

1. Investigar a ausência de `market_value` na base brasileira e comparar o experimento com essa variável removida, em vez de tratá-la como se seu valor fosse conhecido.
2. Avaliar limiares de classificação usando apenas validação americana e então aplicá-los ao Brasil; não escolher o limiar com base nos rótulos brasileiros se o objetivo for medir generalização sem adaptação.
3. Persistir junto aos modelos os parâmetros de pré-processamento e os scores por empresa para tornar a predição reproduzível.
4. Documentar as diferenças entre validação interna americana e validação externa brasileira, incluindo recall, precisão, AUC e matriz de confusão, e revisar as métricas após cada nova execução dos pipelines.