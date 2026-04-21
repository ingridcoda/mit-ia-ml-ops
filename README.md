# 💳 Sistema de Análise de Risco de Crédito (MLOps)

Este projeto implementa um pipeline industrial de Machine Learning para a classificação de risco de crédito, utilizando
o dataset **German Credit (UCI)**. O foco está na modularidade, reprodutibilidade e conformidade com os requisitos do *
*Python 3.13**.

---

## 🛠️ Setup do Ambiente

O projeto utiliza o gerenciador de ambientes nativo do Python. Siga os passos abaixo para configurar sua estação de
trabalho:

### 1. Clonagem e Criação do Virtualenv

Certifique-se de ter o **Python 3.13** instalado no seu sistema.

* **No Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **No Linux/macOS:**
  ```bash
  python3.13 -m venv .venv
  source .venv/bin/activate
  ```

### 2. Instalação de Dependências

Com o ambiente ativo, execute:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuração de Variáveis de Ambiente (.env)

O sistema utiliza um arquivo `.env` para gerenciar chaves da API do Kaggle e configurações de tracking.

1. Localize o arquivo `.env.example` na raiz do projeto.
2. Crie uma cópia chamada `.env`:
   `cp .env.example .env`
3. Preencha suas credenciais do Kaggle (`KAGGLE_USERNAME` e `KAGGLE_KEY`).
   > **Nota:** O arquivo `.env` está no `.gitignore` e nunca deve ser commitado para garantir a segurança das suas
   chaves.

---

## 🏗️ Arquitetura do Sistema

O pipeline foi desenhado seguindo princípios de **Clean Architecture**, dividindo as responsabilidades em módulos
independentes:

| Módulo            | Responsabilidade                                | Tecnologia Chave         |
|:------------------|:------------------------------------------------|:-------------------------|
| **Ingestion**     | Download via API e conversão para Parquet.      | `Kaggle API` / `PyArrow` |
| **Quality**       | Auditoria de dados e conformidade de schema.    | `Great Expectations`     |
| **Preprocessing** | Feature engineering e transformadores autorais. | `Scikit-Learn`           |
| **Modeling**      | Treino, Otimização Bayesiana e MLOps.           | `Optuna` / `MLflow`      |
| **Production**    | Interface de decisão e monitoramento.           | `Streamlit`              |

---

## 🚀 Como Executar

### Pipeline de Dados

O fluxo de dados segue a numeração dos notebooks na pasta `notebooks/`:

1. `01_ingestao_e_qualidade.ipynb`: Coleta os dados e valida a integridade.
2. `02_preprocessamento.ipynb`: Gera o vetor de features final.
3. `03_modelagem.ipynb`: Executa a busca por hiperparâmetros e registra o modelo.

### Aplicação de Produção

Para iniciar a interface de análise de crédito e o dashboard de monitoramento:

```bash
streamlit run production_app/app.py
```

---

## 🧪 Testes Automatizados

A suíte de testes garante que as refatorações não quebrem a lógica de negócio:

```bash
pytest tests/
```

---

## 📈 Tecnologias & Versões

* **Linguagem:** Python 3.13
* **Tracking:** MLflow (Backend SQLite)
* **Algoritmos:** XGBoost / Scikit-Learn
* **Formato de Dados:** Apache Parquet (Snappy Compression)
* **Validação de Dados:** Great Expectations
* **Otimização de Hiperparâmetros:** Optuna
* **Interface de Produção:** Streamlit
* **Gerenciamento de Ambientes:** Virtualenv
* **Dataset:** German Credit (UCI)
* **Autor:** Ingrid Coda
* **Criação:** Junho de 2024
