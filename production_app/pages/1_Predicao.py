import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.model_utils import load_credit_model
from utils.pipeline_utils import preprocessar_entradas

st.set_page_config(page_title="Predição de Risco", page_icon="💳")
st.title("🎯 Análise de Risco de Crédito")

if "historico_predicoes" not in st.session_state:
    st.session_state.historico_predicoes = pd.DataFrame()

# Carregar Modelo e Metadados Dinâmicos
model = load_credit_model()
meta_path = Path(__file__).parent.parent.parent / "outputs" / "modeling" / "model_metadata.json"

# Threshold dinâmico com fallback seguro
threshold_final = 0.50
if meta_path.exists():
    with open(meta_path, "r") as f:
        threshold_final = json.load(f).get('best_threshold', 0.50)

if model is None:
    st.error("Modelo não encontrado! Execute o pipeline.")
else:
    with st.form("dados_cliente"):
        st.subheader("Informações do Proponente")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Idade (anos)", 18, 100, 30)
            duration = st.number_input("Duração do Contrato (meses)", 1, 72, 24)
            job_opts = {
                "Não qualificado / Não residente": 0, "Não qualificado / Residente": 1,
                "Qualificado / Funcionário": 2, "Altamente qualificado / Autônomo / Gestão": 3
            }
            job_label = st.selectbox("Qualificação do Trabalho", options=list(job_opts.keys()), index=2)
            job_value = job_opts[job_label]

        with col2:
            amount = st.number_input("Valor Solicitado (DM)", 100.0, 20000.0, 5000.0)
            checking_opts = {
                "Saldo Negativo (< 0 DM)": "little", "Saldo Baixo (0 a 200 DM)": "moderate",
                "Saldo Alto (>= 200 DM)": "rich", "Não possui / Desconhecido": "NA"
            }
            checking_label = st.selectbox("Status da Conta Corrente", options=list(checking_opts.keys()))
            checking_value = checking_opts[checking_label]

        if st.form_submit_button("Avaliar Risco de Crédito"):
            raw_data = {'Age': age, 'Duration': duration, 'Job': job_value, 'Credit amount': amount,
                        'Checking account': checking_value}

            # Pré-processamento e Inferência
            features_df = preprocessar_entradas(raw_data)
            prob_bad = float(model.predict_proba(features_df)[0][1])

            # Decisão baseada no threshold otimizado do JSON
            prediction_final = 1 if prob_bad > threshold_final else 0

            st.markdown("---")
            st.subheader("Resultado da Análise")
            st.write(f"**Probabilidade de Inadimplência:** {prob_bad:.2%}")
            st.progress(prob_bad)

            if prediction_final == 0:
                st.success(f"✅ **APROVADO** (Risco abaixo do limiar de {threshold_final:.2f})")
            else:
                st.error(f"❌ **NEGADO** (Risco acima do limiar de {threshold_final:.2f})")

            # Explicabilidade
            with st.expander("🔬 Detalhes da Decisão (Feature Importance)"):
                estimador_final = model.named_steps['estimator']
                pesos = estimador_final.feature_importances_

                # Engenharia visual: ajusta a exibição dependendo se houve redução (LDA) ou não
                if len(pesos) == len(features_df.columns):
                    # Se for passthrough (várias colunas), faz o gráfico normal
                    nomes = features_df.columns
                    importancias = pd.Series(pesos, index=nomes)
                    st.bar_chart(importancias.sort_values(ascending=True), horizontal=True)
                else:
                    # Se for LDA (1 componente), exibe um card de aviso em vez de um gráfico feio
                    st.info(
                        "ℹ️ A técnica de redução de dimensionalidade (LDA) consolidou as features originais em um único vetor matemático.")
                    st.metric("Peso do Componente LDA na Decisão", "100%")

            # Histórico
            features_df["Resultado"] = "APROVADO" if prediction_final == 0 else "NEGADO"
            st.session_state.historico_predicoes = pd.concat([features_df, st.session_state.historico_predicoes],
                                                             ignore_index=True).head(10)

    if not st.session_state.historico_predicoes.empty:
        st.subheader("📋 Últimas Análises")
        st.dataframe(st.session_state.historico_predicoes, width='stretch', hide_index=True)
