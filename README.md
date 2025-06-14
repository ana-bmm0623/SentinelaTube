# 🤖 SentinelaTube: Analisador de Sentimentos e Emoções para YouTube

SentinelaTube é uma aplicação web interativa que utiliza múltiplos modelos Transformer para realizar análise profunda de comentários do YouTube em português e inglês, oferecendo uma visão multifacetada da recepção do público.

![Demo da aplicação](https://via.placeholder.com/600x300?text=Demo+da+Aplicação)

## 📋 Tabela de Conteúdos
- [Sobre o Projeto](#-sobre-o-projeto)
- [Tecnologias Utilizadas](#️-tecnologias-utilizadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Guia de Instalação e Uso](#-guia-de-instalação-e-uso)
    - [Treinamento dos Modelos](#etapa-1-treinamento-dos-modelos-no-google-colab)
    - [Executando a Aplicação](#etapa-2-executando-a-aplicação-na-sua-máquina-local)
- [Conexão com o Seminário Acadêmico](#-conexão-com-o-seminário-acadêmico)
- [Melhorias Futuras](#-melhorias-futuras)

## 🎯 Sobre o Projeto

Este projeto demonstra o poder dos modelos de linguagem Transformer em um cenário real. A ferramenta permite analisar comentários de vídeos do YouTube respondendo a três perguntas essenciais:

1. **Qual o sentimento geral?** Classificação em Positivo, Negativo ou Neutro.
2. **Quais as emoções predominantes?** Classificação em 6 categorias (alegria, tristeza, raiva, medo, surpresa, amor).
3. **Sobre o que as pessoas estão falando?** Extração dos principais tópicos e palavras-chave.

O projeto utiliza dois modelos de IA distintos, treinados via Transfer Learning (fine-tuning), trabalhando em conjunto para analisar os comentários.

## 🛠️ Tecnologias Utilizadas

- **🐍 Python 3.10+**
- **🔥 PyTorch**: Framework de deep learning
- **🤗 Transformers (Hugging Face)**: Para modelos BERT e DistilBERT
- **📊 Pandas**: Para manipulação de dados
- **⚙️ Scikit-learn**: Para extração de tópicos e métricas
- **🌐 Streamlit**: Para interface web interativa
- **▶️ Google API Client**: Para API de Dados do YouTube
- **☁️ Google Colab**: Para treinamento com GPU

## 📂 Estrutura do Projeto

```
📁 SentinelaTube/
│
├── 📁 .streamlit/
│   └── 📄 secrets.toml           # <-- (SECRETO) Chave da API do YouTube
│
├── 📁 meu_modelo_emocoes/        # <-- Modelo de Emoções treinado
│   └── ...                      
│
├── 📁 meu_modelo_sentimento/     # <-- Modelo de Sentimento treinado
│   └── ...                      
│
├── 📄 .gitignore                 # <-- Arquivos e pastas ignorados pelo Git
│
├── 📄 README.md                  # <-- Documentação
│
├── 📄 requirements.txt           # <-- Dependências do projeto
│
├── 📄 app.py                     # <-- APLICAÇÃO WEB: Interface Streamlit
│
├── 📄 config.py                  # <-- MÓDULO: Configurações e constantes
│
├── 📄 youtube_api.py             # <-- MÓDULO: Funções da API do YouTube
│
└── 📄 analysis_tools.py          # <-- MÓDULO: Funções de IA e análise
```

> **Nota:** Os scripts de treinamento e dados brutos podem ser mantidos em uma pasta `training/` separada.

## 🚀 Guia de Instalação e Uso

O fluxo de trabalho é dividido: treinamento no Colab e execução local da aplicação.

### Etapa 1: Treinamento dos Modelos (No Google Colab)

1. **Configure o Notebook**: Crie um notebook no Google Colab com T4 GPU
2. **Instale as Dependências**:
     ```bash
     !pip install pandas scikit-learn datasets "transformers[torch]" nltk
     ```
3. **Faça o Upload dos Dados**: Envie os arquivos necessários (youtube_comment_sentiment.csv, train.txt, test.txt, val.txt)
4. **Execute os Scripts de Treinamento**: Para sentimento e emoções
5. **Baixe os Modelos**: Compacte as pastas geradas e faça download

### Etapa 2: Executando a Aplicação (Na sua Máquina Local)

1. **Prepare a Pasta**: Crie a pasta SentinelaTube e descompacte os modelos
2. **Crie os Arquivos**: Prepare os 4 arquivos da aplicação
3. **Configure o Ambiente Local**:
     ```bash
     # Crie e ative um ambiente virtual
     python -m venv venv
     .\venv\Scripts\activate

     # Instale as dependências
     pip install -r requirements.txt

     # Baixe os dados do NLTK
     python -c "import nltk; nltk.download('stopwords')"
     ```
4. **Configure a Chave da API**:
     - Crie `.streamlit/secrets.toml`
     - Adicione: `YOUTUBE_API_KEY = "sua_chave_aqui"`
5. **Execute a Aplicação**:
     ```bash
     streamlit run app.py
     ```

## 🎓 Conexão com o Seminário Acadêmico

Este projeto atende aos critérios de avaliação:

- **Domínio Conceitual**: Aplicação de Transfer Learning sobre modelos Transformer
- **Profundidade Técnica**: Arquitetura dos modelos, parâmetros de fine-tuning, métricas de avaliação
- **Aplicação Contextualizada**: Cenário real de análise de mídias sociais
- **Pensamento Crítico**: Abordagem das limitações (sarcasmo, vieses)

## 🔮 Melhorias Futuras

- **Análise de Sentimento Baseada em Aspectos (ABSA)**: Identificação de sentimento sobre entidades específicas
- **Análise de Tendências**: Monitoramento da evolução do sentimento ao longo do tempo
- **Hospedagem na Nuvem**: Deploy na Streamlit Community Cloud ou Hugging Face Spaces