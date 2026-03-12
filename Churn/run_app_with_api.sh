#!/bin/bash

# Script para executar a aplicação no modo API
# O modelo é servido via MLFlow REST API e o Streamlit consome essa API

echo "🚀 Iniciando Sistema de Churn (Modo API)..."
echo ""
echo "Este script irá iniciar:"
echo "  1. MLFlow UI (porta 5000)"
echo "  2. MLFlow Model Server API (porta 5001)"
echo "  3. Streamlit App (porta 8501)"
echo ""

# Função para cleanup
cleanup() {
    echo ""
    echo "🛑 Encerrando serviços..."
    kill $MLFLOW_UI_PID 2>/dev/null
    kill $MODEL_SERVER_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# 1. Iniciar MLFlow UI
echo "📊 Iniciando MLFlow UI (porta 5000)..."
mlflow ui --port 5000 &
MLFLOW_UI_PID=$!
sleep 2

# 2. Iniciar MLFlow Model Server
echo "🔌 Iniciando MLFlow Model Server API (porta 5001)..."
mlflow models serve -m "models:/churn_best_model/latest" -p 5001 --no-conda &
MODEL_SERVER_PID=$!
echo "⏳ Aguardando API inicializar..."
sleep 5

# Verificar se a API está respondendo
echo "🔍 Verificando API..."
if curl -s http://localhost:5001/ping > /dev/null; then
    echo "✅ API respondendo!"
else
    echo "⚠️  API pode não estar pronta ainda. Aguardando mais um momento..."
    sleep 5
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Serviços iniciados com sucesso!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📡 URLs disponíveis:"
echo "  • MLFlow UI:        http://localhost:5000"
echo "  • MLFlow API:       http://localhost:5001"
echo "  • Streamlit (API):  http://localhost:8501"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# 3. Iniciar Streamlit (versão API)
echo "🌐 Iniciando Streamlit App (modo API)..."
streamlit run app/streamlit_app_api.py

# Cleanup ao finalizar
cleanup
