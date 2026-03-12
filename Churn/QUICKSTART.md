# 🚀 Guia Rápido - Sistema de Churn

## ⚡ Início Rápido (3 passos)

### 1️⃣ Ativar o Ambiente

```bash
conda activate regression
```

### 2️⃣ Verificar Modelo Treinado

Se você já rodou o `train_mlflow.py`, pode pular para o passo 3.

Se não, treine o modelo primeiro:

```bash
cd src
python train_mlflow.py
cd ..
```

⏱️ Tempo estimado: 5-10 minutos

### 3️⃣ Executar a Aplicação

**Opção Rápida** (recomendado):
```bash
./run_streamlit.sh
```

**Opção Completa** (com MLFlow UI):
```bash
./run_app.sh
```

---

## 📱 Acessar a Aplicação

Após executar o script, abra no navegador:

- **Streamlit**: http://localhost:8501
- **MLFlow UI**: http://localhost:5000 (se usou `run_app.sh`)

---

## 🎯 Como Usar

1. **Preencha os dados do cliente** na barra lateral esquerda
2. Clique em **"🔮 Prever Churn"**
3. Visualize:
   - ✅ Probabilidade de churn
   - 📊 Status do cliente
   - 💡 Recomendações automáticas
   - 💵 Análise financeira

---

## 🔄 Comandos Úteis

### Ver experimentos do MLFlow
```bash
mlflow ui --port 5000
```

### Retreinar o modelo
```bash
cd src
python train_mlflow.py
```

### Verificar modelos registrados
Acesse: http://localhost:5000 → "Models"

---

## 🆘 Problemas Comuns

### Erro: "No module named 'mlflow'"
```bash
pip install -r requirements.txt
```

### Erro: "Model not found"
```bash
cd src
python train_mlflow.py
```

### Porta 8501 já em uso
```bash
# Encontrar processo
lsof -i :8501

# Matar processo (substitua PID)
kill -9 <PID>
```

---

## 📊 Exemplo de Uso

**Cenário**: Cliente com alto risco

1. **Dados**:
   - Idade: 45 anos
   - Contrato: Month-to-Month
   - Internet: Fiber Optic
   - Tempo: 3 meses
   - Satisfação: 2/5

2. **Resultado**:
   - Probabilidade: 85%
   - Status: ALTO RISCO
   - Recomendação: Contato imediato

3. **Ação**:
   - Ligar para o cliente
   - Oferecer desconto
   - Verificar problemas

---

## 💡 Dicas

- ✅ Use contratos "Two Year" para clientes de menor risco
- ⚠️ Contratos "Month-to-Month" têm maior churn
- 📈 Clientes com < 6 meses tendem a churnar mais
- 🎯 Satisfação < 3 é um forte indicador de churn

---

**Pronto para começar!** 🚀
