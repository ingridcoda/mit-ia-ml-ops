"""app.py — Main Streamlit application for credit risk prediction."""

import logging
import sys
from pathlib import Path

import mlflow
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("CreditApp")

mlflow.set_tracking_uri("sqlite:///mlruns.db")
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="CreditAI - Governança Bancária", layout="wide", page_icon="🏛️")


def show_home():
    logger.info("🏠 Renderizando Home")
    st.title("🏛️ Plataforma Inteligente de Crédito e Governança MLOps")
    st.markdown("""
    Esta plataforma automatiza o ciclo de decisão de crédito bancário, unindo **Engenharia de Machine Learning** com as melhores práticas de **Governança de Dados**. 
    O sistema processa perfis de clientes para prever o risco de inadimplência, utilizando uma infraestrutura resiliente e auditável.
    """)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Objetivo de Negócio", "Redução de Default",
                  help="Minimizar a aprovação de clientes com perfil de mau pagador (BAD).")
    with c2:
        st.metric("Indicador Chave (KPI)", "F1-Score Superior",
                  help="Equilíbrio entre Precisão e Recall para proteger o capital da instituição.")
    with c3:
        st.metric("Público-Alvo", "Célula de Risco B2B",
                  help="Suporte à decisão para analistas de crédito e comitês de risco.")

    st.markdown("---")

    st.subheader("🚀 Integridade do Ecossistema")
    s1, s2, s3 = st.columns(3)

    modeling_dir = ROOT / "outputs" / "modeling"
    modelo_ok = any(modeling_dir.glob("v*/*.json")) if modeling_dir.exists() else False
    dados_ok = (ROOT / "data" / "features" / "credit_features.parquet").exists()

    with s1:
        if modelo_ok:
            st.success("✅ **Modelagem**: Versões v1, v2 e v3 Prontas")
        else:
            st.error("❌ **Modelagem**: Modelos não localizados")
    with s2:
        if dados_ok:
            st.success("✅ **Feature Store**: Base Parquet Integra")
        else:
            st.error("❌ **Feature Store**: Dados de treino indisponíveis")
    with s3:
        try:
            mlflow.search_experiments()
            st.success("✅ **MLflow**: Conexão de Governança OK")
        except:
            st.warning("⚠️ **MLflow**: Servidor Offline")

    st.info(
        "**Nota Técnica:** Arquitetura baseada em Scikit-Learn Pipelines com persistência via Skops para segurança cibernética contra execução de código arbitrário.")


pg = st.navigation([
    st.Page(show_home, title="Página Inicial", icon="🏠", default=True),
    st.Page("pages/1_Predicao.py", title="Operação: Análise de Crédito", icon="💳"),
    st.Page("pages/2_Monitoramento.py", title="Governança: Painel de Drift", icon="📊")
])

pg.run()

st.markdown("---")
st.markdown(
    "*Desenvolvido por [Ingrid Coda](https://linkedin.com/in/ingridcoda) para fins acadêmicos - 2026*",
    help="Projeto de conclusão da disciplina de MLOps no Instituto Infnet."
)
st.caption("MIT em Inteligência Artificial, Machine Learning e Deep Learning - Infnet 2026")
