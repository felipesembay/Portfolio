import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import os

# Configuração da página
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-high {
        color: #d62728;
        font-weight: bold;
    }
    .prediction-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Carrega o modelo do MLFlow Model Registry"""
    # Configurar tracking URI
    # Navega de app/ para a raiz do projeto
    project_root = Path(__file__).parent.parent
    mlruns_path = project_root / "mlruns"
    
    mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
    
    try:
        # Tentar carregar o modelo registrado
        model_name = "churn_best_model"
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model, "Model Registry"
    except:
        try:
            # Se falhar, tentar carregar do último run
            model = mlflow.sklearn.load_model("runs:/latest/model")
            return model, "Latest Run"
        except:
            return None, None


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
    """Cria o formulário de entrada de dados"""
    st.sidebar.header("📝 Informações do Cliente")
    
    # Informações demográficas
    st.sidebar.subheader("Dados Demográficos")
    gender = st.sidebar.selectbox("Gênero", ["Male", "Female"])
    age = st.sidebar.slider("Idade", 18, 100, 40)
    senior_citizen = "Yes" if age >= 65 else "No"
    under_30 = "Yes" if age < 30 else "No"
    married = st.sidebar.selectbox("Casado(a)", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Possui Dependentes", ["Yes", "No"])
    number_of_dependents = st.sidebar.number_input("Número de Dependentes", 0, 10, 0) if dependents == "Yes" else 0
    
    # Informações de serviço
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
    
    # Informações de contrato e pagamento
    st.sidebar.subheader("Contrato e Pagamento")
    contract = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-Month", "One Year", "Two Year"])
    paperless_billing = st.sidebar.selectbox("Fatura Digital", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Método de Pagamento", 
                                         ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    monthly_charge = st.sidebar.number_input("Cobrança Mensal ($)", 0.0, 200.0, 70.0, 0.01)
    
    # Informações adicionais
    st.sidebar.subheader("Outras Informações")
    referred_a_friend = st.sidebar.selectbox("Indicou um Amigo", ["Yes", "No"])
    number_of_referrals = st.sidebar.number_input("Número de Indicações", 0, 10, 0) if referred_a_friend == "Yes" else 0
    offer = st.sidebar.selectbox("Oferta", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
    
    # Calcular campos derivados
    avg_monthly_long_distance_charges = monthly_charge * 0.15 if phone_service == "Yes" else 0.0
    total_charges = monthly_charge * tenure_in_months
    total_long_distance_charges = avg_monthly_long_distance_charges * tenure_in_months
    total_refunds = 0.0
    total_extra_data_charges = 0.0
    total_revenue = total_charges + total_long_distance_charges
    
    quarter = st.sidebar.selectbox("Trimestre", ["Q1", "Q2", "Q3", "Q4"])
    population = st.sidebar.number_input("População da Cidade", 10000, 1000000, 50000)
    satisfaction_score = st.sidebar.slider("Pontuação de Satisfação", 1, 5, 3)
    
    # Criar dataframe
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
    # Header
    st.markdown('<h1 class="main-header">📊 Sistema de Previsão de Churn</h1>', unsafe_allow_html=True)
    
    # Informações sobre o modelo
    with st.expander("ℹ️ Sobre o Sistema"):
        st.write("""
        Este sistema utiliza **Machine Learning** para prever a probabilidade de um cliente cancelar o serviço (churn).
        
        **Como funciona:**
        1. Preencha as informações do cliente na barra lateral
        2. Clique em "Prever Churn"
        3. Visualize a probabilidade e recomendações
        
        **Modelo:** Treinado com MLFlow e registrado no Model Registry
        """)
    
    # Carregar modelo
    model, source = load_model()
    
    if model is None:
        st.error("❌ Erro ao carregar o modelo. Verifique se o modelo foi treinado e registrado no MLFlow.")
        st.info("Execute o script train_mlflow.py primeiro para treinar o modelo.")
        return
    
    st.success(f"✅ Modelo carregado com sucesso! (Fonte: {source})")
    
    # Criar formulário
    input_data = create_input_form()
    
    # Botão de previsão
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Prever Churn", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analisando dados do cliente..."):
            try:
                # Fazer previsão
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
                churn_score = int(probability * 100)  # Score de 0 a 100
                
                # Resultados
                st.markdown("---")
                st.subheader("📈 Resultados da Análise")
                
                # Métricas principais
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
                        label="Probabilidade de Churn",
                        value=f"{probability * 100:.1f}%"
                    )
                
                # Gauge chart
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
                
                # Explicação do Score
                st.info(f"""
                📊 **Sobre o Score de Churn ({churn_score}/100):**
                
                Este score representa a probabilidade do cliente cancelar o serviço:
                - **0-30**: Baixo risco - Cliente estável e satisfeito
                - **31-70**: Risco moderado - Atenção necessária
                - **71-100**: Alto risco - Ação imediata requerida
                
                Quanto maior o score, maior a urgência de retenção.
                """)
                
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
                
                if expected_value > 0:
                    st.success(f"✅ Vale a pena investir na retenção deste cliente!")
                else:
                    st.info("ℹ️ Avaliar custo-benefício da retenção")
                    
            except Exception as e:
                st.error(f"❌ Erro ao fazer previsão: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Powered by MLFlow + Streamlit | Churn Prediction System v1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
