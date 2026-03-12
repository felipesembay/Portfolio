# 🎯 Opções de Execução do Sistema de Churn

Este projeto oferece **3 modos** diferentes de executar a aplicação. Escolha o que melhor se adequa às suas necessidades:

---

## 📊 Comparação Rápida

| Modo | Script | Descrição | Quando Usar |
|------|--------|-----------|-------------|
| **Básico** | `run_streamlit.sh` | Apenas Streamlit | Testes rápidos, desenvolvimento |
| **Completo** | `run_app.sh` | Streamlit + MLFlow UI | Visualizar experimentos |
| **API** | `run_app_with_api.sh` | Streamlit + MLFlow UI + API | Arquitetura distribuída |

---

## 1️⃣ Modo Básico (Recomendado para Início)

### ✨ Características
- ✅ Mais simples e rápido
- ✅ Modelo carregado diretamente no Streamlit
- ✅ Ideal para demos e testes

### 🚀 Como executar
```bash
./run_streamlit.sh
```

### 🌐 Acesso
- Streamlit: http://localhost:8501

### 📋 Arquivos
- `app/streamlit_app.py`

---

## 2️⃣ Modo Completo (Recomendado para Análise)

### ✨ Características
- ✅ Streamlit para predições
- ✅ MLFlow UI para visualizar experimentos
- ✅ Comparar modelos diferentes
- ✅ Ver métricas e artefatos

### 🚀 Como executar
```bash
./run_app.sh
```

### 🌐 Acesso
- Streamlit: http://localhost:8501
- MLFlow UI: http://localhost:5000

### 📋 Arquivos
- `app/streamlit_app.py`

---

## 3️⃣ Modo API (Recomendado para Produção)

### ✨ Características
- ✅ Arquitetura distribuída
- ✅ Modelo servido como API REST
- ✅ Streamlit consome a API
- ✅ MLFlow UI para gerenciamento
- ✅ Fácil escalonamento

### 🚀 Como executar
```bash
./run_app_with_api.sh
```

### 🌐 Acesso
- Streamlit: http://localhost:8501
- MLFlow API: http://localhost:5001
- MLFlow UI: http://localhost:5000

### 📋 Arquivos
- `app/streamlit_app_api.py`

### 🔌 Testar API Manualmente
```bash
# Health check
curl http://localhost:5001/ping

# Fazer predição
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d @test_input.json
```

---

## 🎨 Arquitetura Visual

### Modo Básico
```
┌─────────────────┐
│   Streamlit     │
│   (porta 8501)  │
│                 │
│  [Modelo Local] │
└─────────────────┘
```

### Modo Completo
```
┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │     │   MLFlow UI     │
│   (porta 8501)  │     │   (porta 5000)  │
│                 │     │                 │
│  [Modelo Local] │     │  [Experimentos] │
└─────────────────┘     └─────────────────┘
```

### Modo API
```
┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│  MLFlow API     │
│   (porta 8501)  │HTTP │   (porta 5001)  │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   MLFlow UI     │
                        │   (porta 5000)  │
                        └─────────────────┘
```

---

## 💡 Qual modo escolher?

### Para Desenvolvimento/Teste
```bash
./run_streamlit.sh
```
- Mais rápido de iniciar
- Fácil de debugar
- Ideal para testes

### Para Demonstração/Apresentação
```bash
./run_app.sh
```
- Mostra capacidades completas
- Visualiza experimentos
- Compara modelos

### Para Produção/Deploy
```bash
./run_app_with_api.sh
```
- Separação de responsabilidades
- Escalável
- Profissional

---

## 🔄 Fluxo de Trabalho Típico

### 1. Desenvolvimento
```bash
# Treinar modelos
cd src
python train_mlflow.py

# Testar localmente
cd ..
./run_streamlit.sh
```

### 2. Análise
```bash
# Ver experimentos e comparar modelos
./run_app.sh

# Acessar MLFlow UI em http://localhost:5000
```

### 3. Deploy
```bash
# Servir modelo via API
./run_app_with_api.sh

# Integrar com outras aplicações
```

---

## 📝 Próximos Passos

Depois de escolher um modo e executar:

1. **Acesse o Streamlit** no navegador
2. **Preencha os dados** do cliente na barra lateral
3. **Clique em "Prever Churn"**
4. **Analise os resultados** e recomendações
5. **Experimente diferentes perfis** de clientes

---

## 🆘 Precisa de Ajuda?

- 📖 **Guia Rápido**: Veja `QUICKSTART.md`
- 📚 **Documentação Completa**: Veja `README.md`
- 🔌 **Guia da API**: Veja `MLFLOW_API.md`

---

**Escolha seu modo e comece! 🚀**
