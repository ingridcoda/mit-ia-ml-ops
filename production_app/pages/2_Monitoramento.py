"""2_Monitoramento.py — Monitoring page for drift detection."""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.monitoring import DriftDetector

logger = logging.getLogger("CreditApp.Monitoramento")
ROOT = Path(__file__).parent.parent.parent.resolve()

st.title("📊 Painel de Governança e Saúde de Modelos")
st.markdown("Monitore a eficácia dos modelos e detecte mudanças de comportamento nos dados de produção.")


def carregar_dados():
    base = ROOT / "outputs" / "modeling"
    hist = {}
    if base.exists():
        for d in base.glob("v*"):
            meta = list(d.glob("*.json"))
            if meta:
                with open(meta[0], "r") as f:
                    data = json.load(f)
                    hist[d.name] = {
                        'accuracy': data.get('accuracy', 0.0),
                        'f1': data.get('f1', 0.0),
                        'roc_auc': data.get('roc_auc', 0.0),
                        'selected_reducer': data.get('selected_reducer', 'N/A').upper()
                    }
    return hist


dados = carregar_dados()

if not dados:
    st.warning("Nenhuma execução registrada no histórico.")
else:
    st.subheader("🏆 Benchmarking Experimental")
    df_metrics = pd.DataFrame.from_dict(dados, orient='index')
    st.dataframe(df_metrics, width='stretch')
    st.bar_chart(df_metrics[['f1', 'roc_auc']])

    st.subheader("🔍 Monitoramento de Data Drift")
    st.markdown("""
    **O que é Drift?** É a alteração na distribuição das características dos clientes. 
    Se o perfil de quem pede crédito mudar (Ex: clientes mais jovens ou montantes maiores), o modelo pode perder precisão. 
    Usamos o teste **KS (Kolmogorov-Smirnov)**: um P-Value < 0.05 indica desvio crítico.
    """)

    feat_path = ROOT / "data" / "features" / "credit_features.parquet"
    if feat_path.exists():
        df_feat = pd.read_parquet(feat_path)
        detector = DriftDetector(df_feat.head(500))
        resultados = detector.detectar_drift(df_feat.tail(500))

        tabs = st.tabs([v.upper() for v in dados.keys()])
        for i, v_name in enumerate(dados.keys()):
            with tabs[i]:
                st.write(f"### Estabilidade das Features: {v_name.upper()}")
                df_res = pd.DataFrame([
                    {"Feature": k, "P-Value": f"{v['p_value']:.4f}",
                     "Status": "🔴 DRIFT DETECTADO" if v['drift_detectado'] else "🟢 ESTÁVEL"}
                    for k, v in resultados.items()
                ])
                st.table(df_res)
                if any(v['drift_detectado'] for v in resultados.values()):
                    st.error(f"🚨 **ALERTA**: Desvio detectado. Recomenda-se re-treinamento da {v_name}.")
    else:
        st.error("Feature Store Parquet não encontrada.")
