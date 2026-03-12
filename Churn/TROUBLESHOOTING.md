# 🔍 Diagnóstico do Problema - MLFlow UI Vazia

## ❌ Problema Identificado

Os modelos não aparecem na UI do MLFlow porque:

### Causa Raiz
1. **Tracking URI Inconsistente**: O código apontava para `/mlruns/` mas salvou em `/src/mlruns/`
2. **Runs Não Registrados**: Os runs foram executados mas não foram salvos no banco de dados SQLite
3. **Model Registry Vazio**: Modelos foram salvos como arquivos, mas não registrados no Registry

### Evidências
```bash
# Banco de dados sem runs
$ sqlite3 mlruns/mlflow.db "SELECT COUNT(*) FROM runs;"
0

# Banco de dados sem modelos registrados  
$ sqlite3 mlruns/mlflow.db "SELECT COUNT(*) FROM model_versions;"
0
```

## ✅ Correções Aplicadas

### 1. Código Atualizado

**Arquivo:** [src/train_mlflow.py](src/train_mlflow.py)

Mudanças:
- ✅ Uso de `Path` para paths absolutos e independentes do diretório de execução
- ✅ Tracking URI configurado corretamente antes de criar experimentos
- ✅ Path consistente entre treinamento e aplicação

**Antes:**
```python
mlflow.set_tracking_uri("file:///home/felipe/Projeto/.../mlruns")  # Path absoluto fixo
data_path = "/home/felipe/Projeto/.../data/telco.csv"  # Path absoluto fixo
```

**Depois:**
```python
project_root = Path(__file__).parent.parent  # Encontra raiz do projeto
mlruns_path = project_root / "mlruns"  # Path relativo
data_path = project_root / "data" / "telco.csv"  # Path relativo
mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")  # URI correto
```

### 2. Streamlit App Atualizado

**Arquivo:** [app/streamlit_app.py](app/streamlit_app.py)

Mudanças:
- ✅ Configura tracking URI antes de carregar modelo
- ✅ Usa paths relativos
- ✅ Alinhado com o train_mlflow.py

### 3. Scripts Utilitários

**Novo:** [run_mlflow_ui.sh](run_mlflow_ui.sh)
- Inicia MLFlow UI no diretório correto
- Garante que aponta para `./mlruns/`

**Novo:** [retrain.sh](retrain.sh)
- Limpa experimentos antigos (com backup)
- Retreina com código corrigido
- Automatiza todo o processo

## 🚀 Como Resolver

### Opção 1: Retreinamento Automático (Recomendado)

```bash
conda activate regression
./retrain.sh
```

Isso vai:
1. Fazer backup dos mlruns antigos
2. Limpar diretórios
3. Retreinar todos os modelos
4. Registrar corretamente no MLFlow

⏱️ **Tempo estimado:** 5-10 minutos

### Opção 2: Retreinamento Manual

```bash
conda activate regression

# 1. Limpar mlruns antigo
rm -rf mlruns
mkdir mlruns

# 2. Treinar
cd src
python train_mlflow.py
cd ..

# 3. Ver resultados
./run_mlflow_ui.sh
```

## 📊 Como Verificar se Funcionou

### 1. Verificar Banco de Dados

```bash
# Deve mostrar > 0 runs
sqlite3 mlruns/mlflow.db "SELECT COUNT(*) FROM runs;"

# Deve mostrar o modelo registrado
sqlite3 mlruns/mlflow.db "SELECT name FROM registered_models;"
```

### 2. Verificar MLFlow UI

```bash
# Iniciar UI
./run_mlflow_ui.sh

# Acessar: http://localhost:5000
```

Você deve ver:
- ✅ Experimento "churn_prediction" com ~36 runs
- ✅ Métricas (accuracy, recall, precision, f1, roc_auc) para cada run
- ✅ Na aba "Models", o modelo "churn_best_model"

### 3. Testar Streamlit

```bash
./run_streamlit.sh

# Acessar: http://localhost:8501
```

Deve carregar sem erros e mostrar:
- ✅ "Modelo carregado com sucesso!"
- ✅ Interface completa funcionando

## 📝 Estrutura Correta Após Retreinamento

```
Churn/
├── mlruns/                          # ✅ Único mlruns na raiz
│   ├── mlflow.db                    # Banco de dados com runs
│   ├── 0/                           # Experimento Default
│   ├── 1/                           # Experimento churn_prediction
│   │   ├── <run_id_1>/             # Run individual
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   ├── params/
│   │   │   └── meta.yaml
│   │   ├── <run_id_2>/
│   │   └── ...
│   └── models/                      # Model Registry
│       └── churn_best_model/
├── src/
│   └── train_mlflow.py             # ✅ Código corrigido
└── app/
    └── streamlit_app.py            # ✅ Código corrigido
```

## 🎯 Benefícios da Correção

1. **Portabilidade**: Código funciona em qualquer máquina
2. **Consistência**: Todos os componentes usam mesmo tracking URI
3. **Organização**: Um único diretório mlruns centralizado
4. **Rastreamento**: Todos os runs e métricas visíveis na UI
5. **Model Registry**: Modelos corretamente registrados e versionados

## 🔄 Como Evitar no Futuro

### ❌ NÃO fazer:
```bash
cd src
python train_mlflow.py  # Pode criar mlruns em src/
```

### ✅ FAZER:
```bash
# Na raiz do projeto
./retrain.sh

# OU
python src/train_mlflow.py  # Executar da raiz
```

## 💡 Dica

Depois de retreinar, você pode comparar modelos na UI do MLFlow:
1. Acesse http://localhost:5000
2. Clique no experimento "churn_prediction"
3. Selecione múltiplos runs
4. Clique em "Compare"
5. Veja gráficos comparativos de métricas

---

**Pronto para retreinar?** Execute: `./retrain.sh`
