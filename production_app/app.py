import sys
from pathlib import Path

import streamlit as st

# ── Bootstrap de Path ────────────────────────────────────────────────────────
root_path = Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Crédito AI - Infnet",
    layout="wide",
    page_icon="🏛️"
)


# ── Função com o conteúdo da Home (Ajustada) ──────────────────────────────────
def show_home():
    st.title("🏛️ Sistema de Análise de Risco de Crédito")

    # Grid de Status do Ambiente
    st.subheader("🚀 Status do Sistema")
    s1, s2, s3 = st.columns(3)

    # Verificação de arquivos cruciais
    modelo_ok = Path("outputs/modeling/credit_model_v1.joblib").exists()
    dados_ok = Path("data/features/credit_features.parquet").exists()

    with s1:
        if modelo_ok:
            st.success("✅ **Modelo de IA**: Pronto")
        else:
            st.error("❌ **Modelo de IA**: Não encontrado")

    with s2:
        if dados_ok:
            st.success("✅ **Base de Dados**: Conectada")
        else:
            st.error("❌ **Base de Dados**: Indisponível")

    with s3:
        st.success("✅ **Pipeline MLOps**: Ativo")

    st.markdown(f"""
    ### Instituto Infnet - Projeto da Disciplina de Operacionalização de Modelos com MLOps
    #### Aluna: [Ingrid Ornellas Coda Sant'Anna Gomes](https://infnet.online/members/ingrid_codahotmail-com-2/)

    Este sistema automatiza o ciclo completo de crédito. 
    Use o menu lateral para navegar entre as funcionalidades de predição e monitoramento.
    """)


# ── Configuração de Navegação ────────────────────────────────────────────────
home_page = st.Page(show_home, title="Página Inicial", icon="🏠", default=True)
pred_page = st.Page("pages/1_Predicao.py", title="Predição de Risco", icon="💳")
moni_page = st.Page("pages/2_Monitoramento.py", title="Monitoramento", icon="📊")

pg = st.navigation([home_page, pred_page, moni_page])

# ── Execução única ───────────────────────────────────────────────────────────
pg.run()

# ── RODAPÉ GLOBAL (Aparecerá em todas as páginas) ─────────────────────────────
st.markdown("---")
st.markdown(
    "*Desenvolvido por [Ingrid Coda](https://linkedin.com/in/ingridcoda) para fins acadêmicos - 2026*",
    help="Projeto de conclusão da disciplina de MLOps no Instituto Infnet."
)
