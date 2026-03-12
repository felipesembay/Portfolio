# 📊 Sistema de Previsão de Churn com MLFlow e Streamlit

Sistema completo de Machine Learning para prever churn de clientes, utilizando MLFlow para rastreamento de experimentos e modelo registry, e Streamlit para interface web interativa.

## 🎯 Funcionalidades

- ✅ Treinamento automatizado de múltiplos modelos com MLFlow
- ✅ Registro e versionamento de modelos no MLFlow Model Registry
- ✅ Interface web interativa para prever churn
- ✅ **Score de Churn (0-100):** Métrica clara e objetiva de risco
- ✅ Análise de probabilidade em tempo real com visualização gauge
- ✅ Recomendações automáticas baseadas no nível de risco
- ✅ Análise financeira de ROI de retenção por cliente

## 📁 Estrutura do Projeto

```
Churn/
├── app/
│   └── streamlit_app.py          # Aplicação Streamlit
├── data/
│   └── telco.csv                  # Dataset
├── src/
│   └── train_mlflow.py            # Script de treinamento com MLFlow
├── mlruns/                        # Artefatos do MLFlow
├── models/                        # Modelos salvos
├── notebooks/                     # Notebooks de exploração
└── run_app.sh                     # Script para iniciar a aplicação
```

## � Score de Churn (0-100)

O sistema calcula um **Score de Churn** de 0 a 100 para cada cliente, facilitando a tomada de decisão:

| Score | Nível de Risco | Ação Recomendada |
|-------|----------------|------------------|
| 🟢 **0-30** | Baixo | Programa de fidelidade e incentivos |
| 🟡 **31-70** | Moderado | Campanhas de engajamento e monitoramento |
| 🔴 **71-100** | Alto | Contato imediato e oferta especial |

**Exemplo:**
- Cliente com Score **85/100** → 85% de probabilidade de cancelar → Ação imediata!
- Cliente com Score **23/100** → 23% de probabilidade de cancelar → Cliente estável

📖 **Documentação completa:** [CHURN_SCORE.md](CHURN_SCORE.md)

## �🚀 Como Usar

### 1. Ativar o ambiente conda

```bash
conda activate regression
```

### 2. Instalar Dependências (se necessário)

```bash
pip install mlflow streamlit plotly scikit-learn pandas numpy imbalanced-learn lightgbm xgboost
```

### 3. Treinar o Modelo

Execute o script de treinamento para treinar todos os modelos e registrar o melhor no MLFlow:

```bash
cd src
python train_mlflow.py
```

Este script irá:
- Treinar 36 combinações de modelos (6 modelos × 6 scalers)
- Usar validação cruzada estratificada
- Registrar métricas no MLFlow
- Salvar o melhor modelo no Model Registry como "churn_best_model"

### 4. Executar a Aplicação

#### Opção A: Script Automatizado

```bash
chmod +x run_app.sh
./run_app.sh
```

#### Opção B: Executar Manualmente

Terminal 1 - MLFlow UI:
```bash
mlflow ui --port 5000
```

Terminal 2 - Streamlit:
```bash
streamlit run app/streamlit_app.py
```

### 5. Acessar as Interfaces

- **MLFlow UI**: http://localhost:5000
  - Visualizar experimentos
  - Comparar modelos
  - Gerenciar Model Registry

- **Streamlit App**: http://localhost:8501
  - Interface de predição
  - Análise de clientes
  - Recomendações de retenção

## 📊 Modelos Treinados

O sistema testa os seguintes modelos:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **LightGBM**
6. **XGBoost**

Com os seguintes scalers:
- None
- StandardScaler
- MinMaxScaler
- RobustScaler
- MaxAbsScaler
- Normalizer

## 🎯 Métricas Utilizadas

- **Accuracy**: Acurácia geral
- **Recall**: Taxa de detecção de churns
- **Precision**: Precisão nas previsões de churn
- **F1-Score**: Média harmônica entre precision e recall
- **ROC AUC**: Área sob a curva ROC (métrica principal)

## 💡 Como Usar a Interface Streamlit

1. **Preencher Dados do Cliente**:
   - Use a barra lateral para inserir informações do cliente
   - Preencha dados demográficos, serviços contratados, e informações de contrato

2. **Prever Churn**:
   - Clique no botão "🔮 Prever Churn"
   - Visualize a probabilidade de churn do cliente

3. **Analisar Resultados**:
   - Veja a probabilidade em formato de gauge
   - Leia as recomendações automáticas
   - Analise o perfil do cliente
   - Verifique a análise financeira de retenção

## 🔧 Integração MLFlow + Streamlit

O Streamlit carrega o modelo diretamente do MLFlow Model Registry:

```python
# Carrega o modelo mais recente registrado
model = mlflow.sklearn.load_model("models:/churn_best_model/latest")
```

Isso garante que a aplicação sempre use a versão mais recente do modelo aprovado.

## 📈 Pipeline de ML

O pipeline completo inclui:

1. **Preprocessamento**:
   - Imputação de valores ausentes
   - Clipping de outliers (quantis)
   - One-Hot Encoding para variáveis categóricas

2. **Balanceamento**:
   - NearMiss (undersampling)

3. **Escalonamento** (opcional):
   - Vários scalers testados

4. **Modelo**:
   - Classificador treinado

## 🎨 Interpretação dos Resultados

### Cores do Gauge:
- 🟢 **Verde (0-30%)**: Cliente estável
- 🟡 **Amarelo (30-70%)**: Risco moderado
- 🔴 **Vermelho (70-100%)**: Alto risco de churn

### Recomendações Automáticas:
- **Alto Risco**: Contato imediato, ofertas especiais
- **Risco Moderado**: Campanhas de engajamento
- **Baixo Risco**: Programa de fidelidade

## 📝 Notas Técnicas

- O modelo usa **StratifiedKFold** com 5 folds para validação cruzada
- Todos os experimentos são rastreados no MLFlow
- O melhor modelo é selecionado baseado no **ROC AUC**
- A aplicação Streamlit tem cache para melhor performance

## 🔄 Retreinamento do Modelo

Para retreinar o modelo com novos dados:

1. Atualize o arquivo `data/telco.csv`
2. Execute novamente `python src/train_mlflow.py`
3. O novo modelo será automaticamente registrado no MLFlow
4. Reinicie a aplicação Streamlit para usar o novo modelo

## 📊 Visualizações Disponíveis

- Gauge de probabilidade de churn
- Métricas de performance
- Perfil do cliente
- Análise financeira
- Recomendações personalizadas

## 🤝 Contribuindo

Para melhorias futuras:
- Adicionar SHAP values na interface
- Implementar batch prediction
- Adicionar monitoramento de drift
- Criar dashboard de métricas de negócio

---

**Desenvolvido com ❤️ usando MLFlow + Streamlit**
