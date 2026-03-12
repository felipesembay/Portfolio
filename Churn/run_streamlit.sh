#!/bin/bash

# Script para executar apenas o Streamlit

echo "🌐 Iniciando Streamlit App..."
echo ""
echo "Aplicação disponível em: http://localhost:8501"
echo ""
echo "💡 Dica: Para visualizar os experimentos MLFlow, execute em outro terminal:"
echo "   mlflow ui --port 5000"
echo ""

streamlit run app/streamlit_app.py
