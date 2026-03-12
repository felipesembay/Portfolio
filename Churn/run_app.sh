#!/bin/bash

# Script para executar a aplicação Streamlit

echo "🚀 Iniciando aplicação de Churn Prediction..."
echo ""
echo "📊 MLFlow UI disponível em: http://localhost:5000"
echo "🌐 Streamlit App disponível em: http://localhost:8501"
echo ""

# Iniciar MLFlow UI em background
echo "Iniciando MLFlow UI..."
mlflow ui --port 5000 &
MLFLOW_PID=$!

# Aguardar um momento para MLFlow iniciar
sleep 3

# Iniciar Streamlit
echo "Iniciando Streamlit..."
streamlit run app/streamlit_app.py

# Quando Streamlit for fechado, fechar MLFlow também
kill $MLFLOW_PID
