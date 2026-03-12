# 🔌 Servindo o Modelo via API MLFlow

Este guia mostra como servir o modelo de churn como uma API REST usando o MLFlow.

## 🚀 Opções de Servir o Modelo

### Opção 1: MLFlow Serve (Recomendado)

Serve o modelo diretamente do MLFlow Model Registry:

```bash
# Servir o modelo registrado (porta 5001)
mlflow models serve -m "models:/churn_best_model/latest" -p 5001 --no-conda
```

### Opção 2: Servir por Run ID

Se preferir servir um run específico:

```bash
# Primeiro, pegue o RUN_ID do MLFlow UI
# Depois execute:
mlflow models serve -m "runs:/<RUN_ID>/model" -p 5001 --no-conda
```

---

## 📡 Testando a API

### 1. Verificar se a API está rodando

```bash
curl http://localhost:5001/ping
```

Resposta esperada: `{}`

### 2. Fazer uma predição

Crie um arquivo `test_input.json`:

```json
{
  "dataframe_split": {
    "columns": [
      "Gender", "Age", "Under 30", "Senior Citizen", "Married", "Dependents",
      "Number of Dependents", "Population", "Quarter", "Referred a Friend",
      "Number of Referrals", "Tenure in Months", "Offer", "Phone Service",
      "Avg Monthly Long Distance Charges", "Multiple Lines", "Internet Service",
      "Internet Type", "Avg Monthly GB Download", "Online Security", "Online Backup",
      "Device Protection Plan", "Premium Tech Support", "Streaming TV",
      "Streaming Movies", "Streaming Music", "Unlimited Data", "Contract",
      "Paperless Billing", "Payment Method", "Monthly Charge", "Total Charges",
      "Total Refunds", "Total Extra Data Charges", "Total Long Distance Charges",
      "Total Revenue", "Satisfaction Score"
    ],
    "data": [
      [
        "Male", 45, "No", "No", "Yes", "No", 0, 50000, "Q3", "No",
        0, 3, "None", "Yes", 10.5, "No", "Yes", "Fiber Optic",
        25, "No", "No", "No", "No", "Yes", "Yes", "No", "No",
        "Month-to-Month", "Yes", "Credit Card", 85.0, 255.0,
        0.0, 0.0, 31.5, 286.5, 2
      ]
    ]
  }
}
```

### 3. Enviar requisição

```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d @test_input.json
```

Resposta esperada:
```json
{"predictions": [1]}
```

---

## 🔗 Integrando com Streamlit

### Opção A: Usar Modelo Diretamente (Atual)

O Streamlit carrega o modelo diretamente do MLFlow:

```python
model = mlflow.sklearn.load_model("models:/churn_best_model/latest")
prediction = model.predict(data)
```

**Vantagens:**
- ✅ Mais rápido
- ✅ Menos dependências
- ✅ Sem necessidade de API externa

**Desvantagens:**
- ❌ Modelo e app no mesmo processo
- ❌ Dificulta escalonamento

### Opção B: Usar API REST

Modificar o Streamlit para chamar a API:

```python
import requests
import json

def predict_via_api(data):
    url = "http://localhost:5001/invocations"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "dataframe_split": {
            "columns": data.columns.tolist(),
            "data": data.values.tolist()
        }
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()["predictions"]
```

**Vantagens:**
- ✅ Separação de responsabilidades
- ✅ Fácil escalonamento
- ✅ Pode usar diferentes versões do modelo

**Desvantagens:**
- ❌ Mais lento (chamada HTTP)
- ❌ Precisa manter API rodando
- ❌ Mais complexo para deploy

---

## 🐳 Deploy com Docker

### 1. Criar imagem Docker do modelo

```bash
# Construir imagem
mlflow models build-docker -m "models:/churn_best_model/latest" -n churn-model

# Executar container
docker run -p 5001:8080 churn-model
```

### 2. Testar container

```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d @test_input.json
```

---

## 🔄 Workflow Completo

### Terminal 1: MLFlow UI
```bash
mlflow ui --port 5000
```

### Terminal 2: Modelo como API
```bash
mlflow models serve -m "models:/churn_best_model/latest" -p 5001 --no-conda
```

### Terminal 3: Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 URLs de Acesso

- **MLFlow UI**: http://localhost:5000
- **Modelo API**: http://localhost:5001
- **Streamlit**: http://localhost:8501

---

## 📊 Monitoramento

### Ver logs da API

A API do MLFlow mostra logs em tempo real:
- Requisições recebidas
- Tempo de predição
- Erros

### Métricas úteis

- **Latência**: Tempo de resposta da API
- **Throughput**: Requisições por segundo
- **Taxa de erro**: % de predições que falharam

---

## 🆘 Troubleshooting

### Erro: "Address already in use"

```bash
# Encontrar processo na porta 5001
lsof -i :5001

# Matar processo
kill -9 <PID>
```

### Erro: "Model not found"

Verifique se o modelo está registrado:
```bash
# Acessar MLFlow UI
mlflow ui

# Ou listar modelos via CLI
mlflow models list
```

### Erro: "conda environment not found"

Use a flag `--no-conda`:
```bash
mlflow models serve -m "models:/churn_best_model/latest" -p 5001 --no-conda
```

---

## 💡 Dicas de Performance

1. **Uso de Cache**: MLFlow cacheia modelos carregados
2. **Batch Predictions**: Envie múltiplas predições em uma requisição
3. **Load Balancer**: Use nginx para distribuir carga entre múltiplas instâncias
4. **Async**: Use workers assíncronos (gunicorn)

---

## 📝 Exemplo Completo de Integração

```python
# streamlit_app_with_api.py

import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://localhost:5001/invocations"

def predict_churn(data):
    """Faz predição via API do MLFlow"""
    payload = {
        "dataframe_split": {
            "columns": data.columns.tolist(),
            "data": data.values.tolist()
        }
    }
    
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=5
        )
        response.raise_for_status()
        return response.json()["predictions"][0]
    except Exception as e:
        st.error(f"Erro na API: {str(e)}")
        return None

# Uso no Streamlit
if st.button("Prever Churn"):
    prediction = predict_churn(input_data)
    if prediction is not None:
        st.write(f"Predição: {prediction}")
```

---

**Pronto para servir seu modelo!** 🚀
