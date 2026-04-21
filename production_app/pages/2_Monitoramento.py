import json
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp  # Necessário para detecção de Drift

st.set_page_config(page_title="Monitoramento ML", page_icon="📈", layout="wide")
st.title("📊 Monitoramento de Performance e Detecção de Drift")
st.markdown("---")


def carregar_metadados():
    meta_path = Path(__file__).parent.parent.parent / "outputs" / "modeling" / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


meta = carregar_metadados()

# ── 1. Painel de KPIs Técnicos ──────────────────────────────────────────────
st.subheader("🎯 Performance do Modelo")
m1, m2, m3, m4 = st.columns(4)
if meta:
    with m1: st.metric("F1-Score Otimizado", f"{meta.get('f1', 0):.4f}")
    with m2: st.metric("Acurácia", f"{meta.get('accuracy', 0):.1%}")
    with m3: st.metric("Recall (Risco)", f"{meta.get('recall', 0):.1%}")
    with m4: st.metric("Threshold Ativo", f"{meta.get('best_threshold', 0.5):.2f}")

st.markdown("---")

# ── 2. Detecção de Drift (Requisito Parte 6) ────────────────────────────────
st.subheader("🚨 Detecção de Drift de Dados (Simulação de Produção)")
base_dir = Path(__file__).parent.parent.parent
feat_path = base_dir / "data" / "features" / "credit_features.parquet"

if feat_path.exists():
    df = pd.read_parquet(feat_path)

    # Simulamos o drift comparando as primeiras 500 linhas (referência)
    # com as últimas 500 (produção atual)
    ref_data = df.head(500)
    curr_data = df.tail(500)

    drift_metrics = []
    for col in ['Age', 'Duration', 'Credit amount']:
        # Teste KS: p-value < 0.05 indica que as distribuições mudaram (Drift)
        stat, p_val = ks_2samp(ref_data[col], curr_data[col])
        drift_metrics.append({
            "Variável": col,
            "P-Value": f"{p_val:.4f}",
            "Status": "🔴 DRIFT DETECTADO" if p_val < 0.05 else "🟢 ESTÁVEL"
        })

    st.table(pd.DataFrame(drift_metrics))
    st.caption("O teste Kolmogorov-Smirnov identifica se o perfil dos novos clientes mudou em relação ao treino.")

st.markdown("---")

# ── 3. Organização em Abas Analíticas ──────────────────────────────────────────
tab_dados, tab_modelo = st.tabs(["📁 Perfil da Base", "🧠 Diagnóstico do Modelo"])

if feat_path.exists():
    with tab_dados:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Distribuição Etária**")
            st.bar_chart(df['Age'].value_counts().sort_index(), color="#4f8bff")
        with c2:
            st.write("**Equilíbrio de Risco**")
            risk_dist = df['Risk'].value_counts().reset_index()
            risk_dist.columns = ['Risk', 'count']
            risk_dist['Color_Hex'] = risk_dist['Risk'].map({'good': '#2ecc71', 'bad': '#e74c3c'})
            st.bar_chart(risk_dist, x='Risk', y='count', color='Color_Hex', horizontal=True)

    with tab_modelo:
        c3, c4 = st.columns(2)
        with c3:
            st.write("**Duração vs Valor (Colorido por Risco)**")
            df['Cor_Risco'] = df['Risk'].map({'good': '#2ecc71', 'bad': '#e74c3c'})
            st.scatter_chart(data=df, x='Duration', y='Credit amount', color='Cor_Risco')
        with c4:
            st.write("**Impacto do Log Transform**")
            st.line_chart(df[['Credit amount', 'log_Credit amount']].head(50))

    st.markdown("---")
    st.subheader("🕵️ Inspeção de Features")
    st.dataframe(df.head(15), width='stretch', hide_index=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📦 Baixar Features (CSV)", csv, "features_monitoramento.csv", "text/csv")
