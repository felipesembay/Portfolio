#!/bin/bash

# Script para iniciar o MLFlow UI apontando para o diretório correto

echo "🚀 Iniciando MLFlow UI..."
echo ""
echo "📁 Usando diretório: $(pwd)/mlruns"
echo ""
echo "🌐 Acesse: http://localhost:5000"
echo ""

cd /home/felipe/Projeto/Portfolio/Portfolio2/Churn

mlflow ui --backend-store-uri ./mlruns --port 5000
