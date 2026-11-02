# Predição de Falência Corporativa com Machine Learning: Análise de Transferibilidade Cross-Market (EUA ➔ Brasil)

Repositório acadêmico do Trabalho de Conclusão de Curso (TCC) desenvolvido na Faculdade de Computação e Informática (FCI) da Universidade Presbiteriana Mackenzie.

## Visão Geral do Projeto

A identificação antecipada de estresse financeiro e risco de insolvência é um desafio crítico para gestores, credores e reguladores. Historicamente baseada em formulações estatísticas lineares (como o Z-Score de Altman e o modelo de Ohlson), a literatura recente aponta a necessidade de modelos não lineares capazes de capturar relações multivariadas e interações assintóticas inerentes aos demonstrativos contábeis.

O objetivo deste projeto é desenvolver, avaliar e comparar modelos preditivos supervisionados de Aprendizado de Máquina (Regressão Logística, Random Forest e XGBoost) treinados sobre dados contábeis norte-americanos (US Company Bankruptcy Prediction Dataset). O pipeline converte variáveis contábeis nominais em razões financeiras adimensionais clássicas (Liquidez, Rentabilidade, Endividamento e Atividade) para testar a transferibilidade (cross-market) do modelo em empresas brasileiras (B3 / CVM) sem a necessidade de rótulos prévios locais.