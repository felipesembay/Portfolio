#!/bin/bash

# Script para limpar MLFlow e retreinar

echo "🧹 Limpando experimentos antigos..."

# Backup dos mlruns antigos (caso necessário)
if [ -d "mlruns_backup" ]; then
    rm -rf mlruns_backup
fi

if [ -d "mlruns" ] && [ "$(ls -A mlruns 2>/dev/null)" ]; then
    mv mlruns mlruns_backup
    echo "✅ Backup criado em mlruns_backup/"
fi

if [ -d "src/mlruns" ] && [ "$(ls -A src/mlruns 2>/dev/null)" ]; then
    rm -rf src/mlruns
    echo "✅ src/mlruns removido"
fi

# Criar diretório mlruns vazio
mkdir -p mlruns

echo ""
echo "🚀 Iniciando retreinamento com configuração corrigida..."
echo ""

# Ativar ambiente e treinar
cd src
python train_mlflow.py

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Treinamento concluído!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Para visualizar os experimentos execute:"
echo "  ./run_mlflow_ui.sh"
echo ""
echo "Para iniciar a aplicação execute:"
echo "  ./run_streamlit.sh"
echo ""
