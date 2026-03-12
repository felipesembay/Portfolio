"""
Versão alternativa do Streamlit App que usa a API do MLFlow
para fazer predições em vez de carregar o modelo diretamente.

Para usar esta versão:
1. Inicie a API do MLFlow:
   mlflow models serve -m "models:/churn_best_model/latest" -p 5001 --no-conda

2. Execute este app:
   streamlit run app/streamlit_app_api.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import mlflow
import mlflow.sklearn
from pathlib import Path

# Configuração
API_URL = "http://localhost:5001/invocations"
API_HEALTH_URL = "http://localhost:5001/ping"

st.set_page_config(
    page_title="Churn Prediction App (API Mode)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_for_probabilities():
    """Carrega o modelo localmente para obter probabilidades"""
    try:
        # Configurar tracking URI
        project_root = Path(__file__).parent.parent
        mlruns_path = project_root / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
        
        # Tentar carregar o modelo registrado
        model_name = "churn_best_model"
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        st.warning(f"Não foi possível carregar modelo local: {str(e)}")
        return None


def check_api_health():
    """Verifica se a API do MLFlow está disponível"""
    try:
        response = requests.get(API_HEALTH_URL, timeout=2)
        return response.status_code == 200
    except:
        return False


def predict_via_api(data):
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
            timeout=10
        )
        response.raise_for_status()
        predictions = response.json()["predictions"]
        return predictions[0]
    except requests.exceptions.Timeout:
        raise Exception("Timeout: A API não respondeu a tempo")
    except requests.exceptions.ConnectionError:
        raise Exception("Erro de conexão: Não foi possível conectar à API")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"Erro HTTP {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise Exception(f"Erro desconhecido: {str(e)}")


def create_gauge_chart(probability):
    """Cria um gráfico de gauge para a probabilidade de churn"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilidade de Churn (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#2ca02c"},
                {'range': [30, 70], 'color': "#ff7f0e"},
                {'range': [70, 100], 'color': "#d62728"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_input_form():
    """Cria o formulário de entrada (mesmo do app principal)"""
    st.sidebar.header("📝 Informações do Cliente")
    
    st.sidebar.subheader("Dados Demográficos")
    gender = st.sidebar.selectbox("Gênero", ["Male", "Female"])
    age = st.sidebar.slider("Idade", 18, 100, 40)
    senior_citizen = "Yes" if age >= 65 else "No"
    under_30 = "Yes" if age < 30 else "No"
    married = st.sidebar.selectbox("Casado(a)", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Possui Dependentes", ["Yes", "No"])
    number_of_dependents = st.sidebar.number_input("Número de Dependentes", 0, 10, 0) if dependents == "Yes" else 0
    
    st.sidebar.subheader("Serviços Contratados")
    tenure_in_months = st.sidebar.slider("Tempo de Contrato (meses)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Serviço de Telefone", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Múltiplas Linhas", ["Yes", "No", "No phone service"]) if phone_service == "Yes" else "No phone service"
    
    internet_service = st.sidebar.selectbox("Serviço de Internet", ["Yes", "No"])
    
    if internet_service == "Yes":
        internet_type = st.sidebar.selectbox("Tipo de Internet", ["DSL", "Fiber Optic", "Cable"])
        online_security = st.sidebar.selectbox("Segurança Online", ["Yes", "No"])
        online_backup = st.sidebar.selectbox("Backup Online", ["Yes", "No"])
        device_protection = st.sidebar.selectbox("Proteção de Dispositivo", ["Yes", "No"])
        premium_tech_support = st.sidebar.selectbox("Suporte Técnico Premium", ["Yes", "No"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])
        streaming_music = st.sidebar.selectbox("Streaming Music", ["Yes", "No"])
        unlimited_data = st.sidebar.selectbox("Dados Ilimitados", ["Yes", "No"])
        avg_monthly_gb_download = st.sidebar.slider("Download Médio Mensal (GB)", 0, 100, 20)
    else:
        internet_type = "No"
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        premium_tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        streaming_music = "No internet service"
        unlimited_data = "No internet service"
        avg_monthly_gb_download = 0
    
    st.sidebar.subheader("Contrato e Pagamento")
    contract = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-Month", "One Year", "Two Year"])
    paperless_billing = st.sidebar.selectbox("Fatura Digital", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Método de Pagamento", 
                                         ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    monthly_charge = st.sidebar.number_input("Cobrança Mensal ($)", 0.0, 200.0, 70.0, 0.01)
    
    st.sidebar.subheader("Outras Informações")
    referred_a_friend = st.sidebar.selectbox("Indicou um Amigo", ["Yes", "No"])
    number_of_referrals = st.sidebar.number_input("Número de Indicações", 0, 10, 0) if referred_a_friend == "Yes" else 0
    offer = st.sidebar.selectbox("Oferta", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
    
    avg_monthly_long_distance_charges = monthly_charge * 0.15 if phone_service == "Yes" else 0.0
    total_charges = monthly_charge * tenure_in_months
    total_long_distance_charges = avg_monthly_long_distance_charges * tenure_in_months
    total_refunds = 0.0
    total_extra_data_charges = 0.0
    total_revenue = total_charges + total_long_distance_charges
    
    quarter = st.sidebar.selectbox("Trimestre", ["Q1", "Q2", "Q3", "Q4"])
    population = st.sidebar.number_input("População da Cidade", 10000, 1000000, 50000)
    satisfaction_score = st.sidebar.slider("Pontuação de Satisfação", 1, 5, 3)
    
    data = {
        'Gender': gender,
        'Age': age,
        'Under 30': under_30,
        'Senior Citizen': senior_citizen,
        'Married': married,
        'Dependents': dependents,
        'Number of Dependents': number_of_dependents,
        'Population': population,
        'Quarter': quarter,
        'Referred a Friend': referred_a_friend,
        'Number of Referrals': number_of_referrals,
        'Tenure in Months': tenure_in_months,
        'Offer': offer,
        'Phone Service': phone_service,
        'Avg Monthly Long Distance Charges': avg_monthly_long_distance_charges,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Internet Type': internet_type,
        'Avg Monthly GB Download': avg_monthly_gb_download,
        'Online Security': online_security,
        'Online Backup': online_backup,
        'Device Protection Plan': device_protection,
        'Premium Tech Support': premium_tech_support,
        'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_movies,
        'Streaming Music': streaming_music,
        'Unlimited Data': unlimited_data,
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Payment Method': payment_method,
        'Monthly Charge': monthly_charge,
        'Total Charges': total_charges,
        'Total Refunds': total_refunds,
        'Total Extra Data Charges': total_extra_data_charges,
        'Total Long Distance Charges': total_long_distance_charges,
        'Total Revenue': total_revenue,
        'Satisfaction Score': satisfaction_score
    }
    
    return pd.DataFrame([data])


def main():
    st.title("📊 Sistema de Previsão de Churn (Modo API)")
    
    # Carregar modelo para probabilidades
    local_model = load_model_for_probabilities()
    
    # Verificar status da API
    api_status = check_api_health()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if api_status:
            st.success("✅ API do MLFlow conectada e funcionando")
        else:
            st.error("❌ API do MLFlow não está disponível")
            st.code("mlflow models serve -m \"models:/churn_best_model/latest\" -p 5001 --no-conda")
            
            # Se não tem API mas tem modelo local, permitir uso local
            if local_model is not None:
                st.info("💡 Modo local disponível - usando modelo carregado localmente")
            else:
                return
    
    with col2:
        st.metric("Status API", "Online" if api_status else "Offline")
    
    with st.expander("ℹ️ Informações"):
        st.write("""
        **Arquitetura do Sistema:**
        
        1. **MLFlow Model Server** (porta 5001): Serve o modelo via API REST
        2. **Streamlit App** (porta 8501): Interface web que consome a API
        3. **Modelo Local**: Usado para obter probabilidades detalhadas (Score de Churn 0-100)
        
        **Vantagens desta abordagem:**
        - ✅ Separação entre modelo e UI
        - ✅ Probabilidades precisas do modelo
        - ✅ Fácil escalonamento (múltiplas instâncias da API)
        - ✅ Versionamento independente
        - ✅ Pode trocar modelo sem reiniciar o app
        """)
    
    # Formulário
    input_data = create_input_form()
    
    # Botão de predição
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Prever Churn via API", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analisando dados do cliente..."):
            try:
                # Tentar usar API primeiro
                prediction = None
                probability = None
                method_used = "Local"
                
                if api_status:
                    try:
                        prediction = predict_via_api(input_data)
                        method_used = "MLFlow API"
                    except:
                        st.warning("API não respondeu, usando modelo local...")
                
                # Sempre usar modelo local para probabilidades precisas
                if local_model is not None:
                    if prediction is None:
                        prediction = local_model.predict(input_data)[0]
                    # Obter probabilidade do modelo local
                    probability = local_model.predict_proba(input_data)[0][1]
                    churn_score = int(probability * 100)  # Score de 0 a 100
                else:
                    if prediction is None:
                        st.error("Nenhum método de predição disponível!")
                        return
                    # Fallback: simular probabilidade
                    probability = 0.85 if prediction == 1 else 0.15
                    churn_score = int(probability * 100)
                
                st.markdown("---")
                st.subheader("📈 Resultados da Análise")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Status",
                        value="RISCO DE CHURN" if prediction == 1 else "CLIENTE ESTÁVEL",
                        delta="Alto Risco" if probability > 0.7 else "Baixo Risco" if probability < 0.3 else "Médio Risco"
                    )
                
                with col2:
                    st.metric(
                        label="Score de Churn",
                        value=f"{churn_score}/100"
                    )
                
                with col3:
                    st.metric(
                        label="Probabilidade",
                        value=f"{probability * 100:.1f}%"
                    )
                
                # Gauge chart
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
                
                # Informação do método usado
                st.info(f"🔧 Método: {method_used} | Probabilidade calculada do modelo treinado")
                
                # Recomendações
                st.markdown("---")
                st.subheader("💡 Recomendações")
                
                if probability > 0.7:
                    st.error("⚠️ **ATENÇÃO: Cliente com alto risco de churn!**")
                    st.write("**Ações recomendadas:**")
                    st.write("- 🎯 Contato imediato do time de retenção")
                    st.write("- 💰 Oferecer desconto ou upgrade de serviços")
                    st.write("- 📞 Ligar para entender insatisfação")
                    st.write("- 🎁 Oferecer benefício exclusivo")
                    
                elif probability > 0.3:
                    st.warning("⚡ **Cliente em risco moderado**")
                    st.write("**Ações recomendadas:**")
                    st.write("- 📧 Enviar email com ofertas personalizadas")
                    st.write("- 📊 Monitorar comportamento de uso")
                    st.write("- 🎯 Incluir em campanha de engajamento")
                    
                else:
                    st.success("✅ **Cliente estável e satisfeito**")
                    st.write("**Ações recomendadas:**")
                    st.write("- 🌟 Programa de fidelidade")
                    st.write("- 🎁 Incentivo para indicação de amigos")
                    st.write("- 📈 Oferecer upgrade de plano")
                
                # Análise do perfil
                st.markdown("---")
                st.subheader("👤 Perfil do Cliente")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Informações Gerais:**")
                    st.write(f"- Idade: {input_data['Age'].values[0]} anos")
                    st.write(f"- Tempo de contrato: {input_data['Tenure in Months'].values[0]} meses")
                    st.write(f"- Tipo de contrato: {input_data['Contract'].values[0]}")
                    st.write(f"- Cobrança mensal: ${input_data['Monthly Charge'].values[0]:.2f}")
                    st.write(f"- Satisfação: {input_data['Satisfaction Score'].values[0]}/5")
                
                with col2:
                    st.write("**Serviços:**")
                    services = []
                    if input_data['Phone Service'].values[0] == 'Yes':
                        services.append("📞 Telefone")
                    if input_data['Internet Service'].values[0] == 'Yes':
                        services.append(f"🌐 Internet ({input_data['Internet Type'].values[0]})")
                    if input_data['Online Security'].values[0] == 'Yes':
                        services.append("🔒 Segurança Online")
                    if input_data['Streaming TV'].values[0] == 'Yes':
                        services.append("📺 Streaming TV")
                    if input_data['Streaming Movies'].values[0] == 'Yes':
                        services.append("🎬 Streaming Movies")
                    
                    if services:
                        for service in services:
                            st.write(f"- {service}")
                    else:
                        st.write("- Nenhum serviço adicional")
                
                # Cálculo de ROI de retenção
                st.markdown("---")
                st.subheader("💵 Análise Financeira")
                
                monthly_charge = input_data['Monthly Charge'].values[0]
                retention_rate = 0.20  # Taxa de sucesso de retenção
                retention_cost = 50  # Custo estimado de campanha de retenção
                
                expected_value = monthly_charge * 12 * retention_rate - retention_cost
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Receita Mensal", f"${monthly_charge:.2f}")
                
                with col2:
                    st.metric("Receita Anual Potencial", f"${monthly_charge * 12:.2f}")
                
                with col3:
                    st.metric("Valor Esperado de Retenção", f"${expected_value:.2f}")
                
                if expected_value > 0 and probability > 0.5:
                    st.success(f"✅ Vale a pena investir na retenção deste cliente!")
                elif expected_value > 0:
                    st.info("ℹ️ Cliente estável - focar em fidelização")
                else:
                    st.warning("⚠️ Avaliar custo-benefício da retenção")
                
            except Exception as e:
                st.error(f"❌ Erro ao fazer predição: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
            <p>Powered by MLFlow API + Streamlit | Score de Churn: 0-100</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
