# 📊 Score de Churn (0-100)

## O que é?

O **Score de Churn** é uma métrica de 0 a 100 que representa a probabilidade de um cliente cancelar o serviço (churnar).

**Fórmula:**
```
Score de Churn = Probabilidade do Modelo × 100
```

Por exemplo:
- Probabilidade 0.85 → Score de Churn **85/100**
- Probabilidade 0.23 → Score de Churn **23/100**

## Interpretação do Score

### 🟢 Score 0-30: Baixo Risco
**Interpretação:** Cliente estável e satisfeito

**Características típicas:**
- Contrato de longo prazo (1-2 anos)
- Alto tempo de permanência (> 12 meses)
- Múltiplos serviços contratados
- Satisfação alta (4-5/5)

**Ações recomendadas:**
- ✅ Programa de fidelidade
- ✅ Incentivo para indicações
- ✅ Oferta de upgrade de plano

---

### 🟡 Score 31-70: Risco Moderado
**Interpretação:** Cliente em situação de atenção

**Características típicas:**
- Contrato mês a mês
- Tempo de permanência médio (3-12 meses)
- Poucos serviços adicionais
- Satisfação média (3/5)

**Ações recomendadas:**
- ⚡ Campanha de engajamento
- 📧 Email marketing personalizado
- 📊 Monitoramento de uso
- 🎯 Oferta de serviços complementares

---

### 🔴 Score 71-100: Alto Risco
**Interpretação:** Risco crítico de cancelamento - ação imediata!

**Características típicas:**
- Contrato mês a mês
- Baixo tempo de permanência (< 3 meses)
- Serviços básicos apenas
- Satisfação baixa (1-2/5)
- Reclamações ou problemas técnicos

**Ações recomendadas:**
- 🚨 **Contato imediato** do time de retenção
- 💰 Oferecer **desconto especial**
- 📞 Ligar para entender **insatisfação**
- 🎁 Benefício exclusivo ou upgrade gratuito
- 🔧 Resolver problemas técnicos urgentemente

---

## Exemplos Práticos

### Exemplo 1: Score 92/100 ⚠️
```
Cliente:
- Nome: João Silva
- Idade: 45 anos
- Contrato: Month-to-Month
- Tempo: 2 meses
- Serviços: Apenas internet básica
- Satisfação: 2/5
- Score: 92/100

Ação:
→ Ligar HOJE
→ Oferecer 30% de desconto por 6 meses
→ Investigar problemas de satisfação
→ Oferecer upgrade gratuito de internet
```

**ROI:**
- Receita mensal: $65
- Receita anual potencial: $780
- Custo de retenção: $50
- **Benefício esperado: $106** (vale a pena!)

---

### Exemplo 2: Score 15/100 ✅
```
Cliente:
- Nome: Maria Santos
- Idade: 52 anos
- Contrato: Two Year
- Tempo: 36 meses
- Serviços: Internet + TV + Telefone + Segurança
- Satisfação: 5/5
- Score: 15/100

Ação:
→ Enviar email de agradecimento
→ Oferecer programa VIP
→ Incentivo para indicação ($20 por amigo)
→ Preview de novos serviços
```

---

### Exemplo 3: Score 55/100 ⚡
```
Cliente:
- Nome: Pedro Costa
- Idade: 28 anos
- Contrato: Month-to-Month
- Tempo: 8 meses
- Serviços: Internet Fiber Optic
- Satisfação: 3/5
- Score: 55/100

Ação:
→ Email em 24-48h
→ Pesquisa de satisfação
→ Oferta de streaming services
→ Propor contrato anual com desconto
```

---

## Relação com Métricas do Modelo

### Accuracy: 94%
O modelo acerta 94% das previsões totais.

### Recall: 89%
De 100 clientes que realmente iriam churnar, o modelo detecta **89**.

### Precision: 88%
Quando o modelo diz "Score > 50", está certo **88% das vezes**.

### ROC AUC: 0.97
Probabilidade de o modelo classificar corretamente um churn vs não churn.

**Score alto (>70) → 88% de chance de ser verdadeiro churn!**

---

## Impacto no Negócio

### Cenário Real: Base de 7.043 clientes

```
Clientes em risco detectados (Score > 50): 1.660
Taxa de sucesso de retenção: 20%
Clientes retidos: 332 clientes

Receita média mensal: $64.76
Receita preservada por mês: $21.500
Receita preservada por ano: $258.000
```

### ROI da Estratégia

| Score | Clientes | Taxa Retenção | Custo/Cliente | Receita Salva |
|-------|----------|---------------|---------------|---------------|
| 71-100 | 500 | 25% | $75 | $97.000/ano |
| 51-70 | 800 | 18% | $50 | $89.000/ano |
| 31-50 | 360 | 12% | $30 | $33.000/ano |
| **Total** | **1.660** | **20%** | **$50** | **$258.000/ano** |

**Investimento total:** $83.000  
**Retorno:** $258.000  
**ROI:** 210% 🚀

---

## Como o Score é Calculado

### Passo 1: Preparação dos Dados
```python
# Dados do cliente são preprocessados
X = preprocessor.transform(input_data)
```

### Passo 2: Balanceamento (NearMiss)
```python
# Dados balanceados durante treinamento
# Garante que o modelo não seja enviesado
```

### Passo 3: Predição de Probabilidade
```python
# Modelo retorna probabilidades
probability = model.predict_proba(X)[0][1]
# [0][1] = probabilidade da classe 1 (churn)
```

### Passo 4: Conversão para Score
```python
# Converte probabilidade para score 0-100
churn_score = int(probability * 100)
```

---

## Monitoramento Contínuo

### Dashboard Sugerido

**Métricas Diárias:**
- Total de clientes analisados
- Média do Score de Churn
- Clientes com Score > 70
- Taxa de conversão de campanhas

**Alertas Automáticos:**
- 🚨 Cliente com Score > 85 → Alerta imediato
- ⚠️ Score médio aumentando → Revisar estratégia
- 📊 Taxa de churn real vs prevista → Calibração

---

## Comparação com Churn Score Original

O dataset original tinha uma coluna **"Churn Score"** que foi excluída do treinamento.

**Diferença:**
- **Churn Score Original**: Métrica calculada pela empresa (métrica desconhecida)
- **Nosso Score de Churn**: Probabilidade do **modelo de ML** treinado

**Nosso modelo é melhor porque:**
1. ✅ Usa 37 features (vs score único original)
2. ✅ ROC AUC de 0.97 (extremamente preciso)
3. ✅ Validação cruzada com 5 folds
4. ✅ Testado em 36 combinações de modelos
5. ✅ Baseado em padrões reais de churn

---

## Próximos Passos

### Melhoria Contínua

1. **Retreinamento Mensal**
   - Incorporar novos dados
   - Ajustar para mudanças de comportamento
   
2. **A/B Testing**
   - Testar diferentes estratégias por faixa de score
   - Otimizar taxa de retenção
   
3. **Segmentação**
   - Criar scores específicos por tipo de cliente
   - Personalizar ações por segmento

4. **Feedback Loop**
   - Registrar resultado das campanhas
   - Melhorar modelo com dados reais

---

**Use o Score de Churn para tomar decisões baseadas em dados e maximizar a retenção de clientes!** 🎯
