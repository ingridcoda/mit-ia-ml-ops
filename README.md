# 🏛️ Sistema Inteligente de Análise de Risco de Crédito (MLOps)

Este projeto implementa um pipeline industrial de Machine Learning para a classificação de risco de crédito, utilizando
o dataset *
*[Statlog (German Credit Data) da UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
**. O foco está na transição da exploração de dados para a **engenharia de machine learning**, garantindo modularidade,
rastreabilidade, controle de dimensionalidade e monitoramento contínuo em produção.

---

## 🏗️ 1. Estrutura de Diretórios e Arquitetura do Sistema

O projeto adota os princípios de **Clean Architecture**, isolando a lógica de negócio, a orquestração e a aplicação
final em pastas distintas:

```text
├── .github/workflows/         # Pipeline de CI/CD (GitHub Actions)
├── notebooks/                 # Scripts executáveis do pipeline (Substituem notebooks)
│   ├── executa_tudo.py        # Orquestrador central parametrizado
│   ├── ingestao.py            # Coleta de dados via API Kaggle
│   ├── modelagem.py           # Otimização Bayesiana e integração MLflow
│   ├── preprocessamento.py    # Feature Engineering e transformadores
│   └── qualidade.py           # Validação via Great Expectations
├── production_app/            # Aplicação de Produção (Streamlit)
│   ├── pages/                 # Telas de Predição e Governança/Drift
│   └── app.py                 # Ponto de entrada do Dashboard
├── src/                       # Core da Lógica de Negócio (Módulos Base)
│   ├── modeling/              # Optuna, MLflowTracker e Redutores
│   ├── monitoring/            # Teste KS (Kolmogorov-Smirnov) para Drift
│   ├── preprocessing/         # Scikit-Learn Custom Transformers
│   └── quality/               # Definição de contratos de dados
└── tests/                     # Suíte de 79 testes automatizados (Pytest)
```

---

## 🛠️ 2. Setup do Ambiente

O projeto foi desenhado para ser totalmente reproduzível em ambientes **[Python 3.13](https://docs.python.org/3.13/)**.

### 2.1 Configuração de Chaves e Variáveis de Ambiente

Crie um arquivo `.env` baseado no `.env.example`:

```env
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave_api
MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

### 2.2 Instalação Local

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

---

## 🚀 3. Guia de Execução

### Passo 1: Geração de Modelos (Pipeline Parametrizado)

Execute o pipeline escolhendo a técnica de redução de dimensionalidade:

```bash
python notebooks/executa_tudo.py --reducer passthrough  # Baseline
python notebooks/executa_tudo.py --reducer pca          # Variância
python notebooks/executa_tudo.py --reducer lda          # Separabilidade
```

### Passo 2: Governança (MLflow)

Abra a interface para auditar experimentos e o Model Registry:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000
```

### Passo 3: Operação e Monitoramento (Streamlit)

```bash
streamlit run production_app/app.py
```

### Passo 4: Validação Técnica (Pytest)

```bash
pytest tests/ -v
```

---

## 📦 4. Infraestrutura: Docker & CI/CD

### 4.1 Docker (Conteinerização)

O projeto inclui um `Dockerfile` baseado em `python:3.13-slim`. Ele garante que a aplicação rode em um ambiente isolado,
idêntico ao de desenvolvimento, mitigando o erro "na minha máquina funciona".

* **Segurança:** A imagem utiliza um usuário não-root (`appuser`) e realiza o scan de vulnerabilidades.
* **Como usar:**
  ```bash
  docker build -t credit-app .
  docker run -p 8501:8501 credit-app
  ```

### 4.2 GitHub Actions (CI/CD)

O arquivo `mlops_pipeline.yaml` automatiza o ciclo de vida:

1. **Linting:** Verifica padrões de código (Flake8).
2. **Testes:** Executa os 79 testes unitários e de integração.
3. **Build:** Constrói a imagem Docker.
4. **Segurança:** Roda o [Trivy](https://github.com/aquasecurity/trivy) para buscar vulnerabilidades na imagem antes do
   deploy.

---

## 📑 5. Relatório Técnico e Decisões de Arquitetura

### 5.1 O Papel da Engenharia e Estruturação

Este projeto consolida a transição de um cenário exploratório para um **projeto orientado à entrega contínua (Engenharia
de ML)**. Reduzimos a dependência de notebooks monolíticos, organizando o código em módulos `.py` escaláveis. O objetivo
técnico é fornecer um sistema resiliente, alinhado à **métrica de negócio (Redução de Default)** e guiado pela **métrica
técnica de F1-Score** (Otimização Bayesiana via Optuna), equilibrando o risco de crédito e a aprovação de bons clientes.
Como Engenheira de ML, as escolhas foram justificadas não apenas pelo desempenho, mas pela viabilidade de operação e
facilidade de manutenção.

### 5.2 Fundação de Dados, Pipelines Leak-Free e Dimensionalidade

Identificamos limitações estruturais e ruídos no dataset (ex: dados ausentes e desbalanceamento), que impactam a
generalização. Para mitigar o *Data Leakage*, modelamos pipelines com **Scikit-Learn**, garantindo que imputações e
transformações ocorram estritamente dentro da validação cruzada.
Avaliamos a redução de dimensionalidade: o **LDA** demonstrou alta capacidade discriminativa com baixo custo
computacional; o **PCA** reteve a variância matemática, mas sacrificou a interpretabilidade. A escolha da abordagem
ideal leva em conta o *trade-off* entre ganho de performance e a necessidade regulatória de explicabilidade (Baseline
Passthrough).

### 5.3 Experimentação, MLflow e Operacionalização

Realizamos experimentos comparativos estruturados, com Validação Cruzada (CV) e Optuna. O **MLflow** garantiu a
rastreabilidade absoluta (parâmetros, F1-Score, instâncias de modelos). A análise crítica dos runs determinou a
assinatura automática da tag `production` para a melhor abordagem.
O modelo vencedor foi persistido de forma versionada (via `skops`) e exposto em um fluxo de inferência funcional via *
*Streamlit**.

### 5.4 Monitoramento e Ciclo de Vida (Drift)

Acompanhar a performance em produção é crítico. O painel Streamlit expõe as métricas e realiza detecção contínua de *
*Data Drift** utilizando o teste estatístico **Kolmogorov-Smirnov (KS)**. Alterações no perfil de entrada (ex: idade,
montante de crédito) disparam alertas automáticos recomendando estratégias de re-treinamento, fechando o ciclo de vida
do modelo.

---

## ✅ 6. Matriz de Conformidade (Rubrica de Avaliação)

Abaixo detalhamos como os critérios avaliativos foram integralmente cumpridos nesta entrega:

| Critério da Rubrica                               | Como e Onde foi Atendido                                                                                                              |
|:--------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| **1.1. Contexto, objetivos e métricas**           | Contexto B2B, objetivo técnico (F1-Score) e negócio (Redução de Default) definidos na interface Streamlit (`app.py`) e Relatório 5.1. |
| **1.2. Exploração vs. Entrega**                   | Foco em entrega evidenciado pelo uso de CI/CD, testes automatizados e empacotamento Docker (Relatório 5.1).                           |
| **1.3. Estruturação de Código**                   | Dependência de notebooks eliminada; uso de scripts orquestradores parametrizados e POO na pasta `src/`.                               |
| **1.4. Decisão Técnica e Viabilidade**            | Escolha de arquitetura pautada no *trade-off* entre performance (Optuna), segurança (Skops) e explicabilidade (Relatório 5.1 e 5.2).  |
| **2.1. Problemas de Dados e Impactos**            | Validação de schema e anomalias implementada com *Great Expectations* (`qualidade.py`), documentado no Relatório 5.2.                 |
| **2.2. Pipelines Scikit-Learn Leak-Free**         | Transformadores customizados agrupados em `Pipeline` do Scikit-Learn, rodando dentro do CV (`src/modeling/model_factory.py`).         |
| **2.3. Implementação de Redução (Justificada)**   | Integração dinâmica de PCA e LDA via orquestrador (`--reducer`), justificados pelo controle de complexidade (Relatório 5.2).          |
| **2.4. Impacto da Redução (Custo/Interpretação)** | Análise documentada no Relatório 5.2 (LDA superior em separabilidade; PCA penalizando a auditoria/interpretabilidade).                |
| **3.1. Planejamento Comparativo**                 | Estrutura parametrizada permitindo comparação 1:1 entre Baseline, PCA e LDA (`notebooks/executa_tudo.py`).                            |
| **3.2. Validação Cruzada e Hiperparâmetros**      | Otimização Bayesiana utilizando `Optuna` com validação cruzada estratificada (`src/modeling/optimizer.py`).                           |
| **3.3. Rastreamento com MLflow**                  | `MLflowTracker` registra parâmetros, métricas e artefatos, mantendo versionamento no `mlruns.db`.                                     |
| **3.4. Análise e Seleção Final**                  | Seleção técnica garantida pela avaliação do F1-Score médio nos *folds*, com promoção automática via script (Relatório 5.3).           |
| **4.1. Persistência Consistente**                 | Modelos salvos serializados via biblioteca segura `skops`, garantindo paridade treino/inferência (`outputs/`).                        |
| **4.2. Fluxo de Inferência Funcional**            | Modelo consumido por serviço de front-end preditivo em `production_app/pages/1_Predicao.py`.                                          |
| **4.3. Métricas em Operação**                     | Definição clara de monitoramento técnico (Acurácia/F1/ROC) e visualizações de negócio no Dashboard Streamlit.                         |
| **4.4. Operação, Drift e Re-treinamento**         | Módulo `monitoring/` executa teste KS para Data Drift, indicando degradação e necessidade de re-treinamento na aba "Monitoramento".   |

---
**Desenvolvido por [Ingrid Coda](mailto:ingrid.gomes@al.infnet.edu.br) para fins acadêmicos em Abril/2026**

**MIT em IA, ML e Deep Learning - Infnet**
