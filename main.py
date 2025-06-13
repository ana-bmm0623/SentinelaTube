# main.py - SCRIPT PARA TREINAR E ENVIAR O MODELO PARA O HUB
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT FINAL PARA TREINAMENTO DO MODELO DE ANÁLISE DE SENTIMENTOS
Este script tem a única função de treinar o modelo e enviá-lo para o
Hugging Face Hub.
================================================================================
"""
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import login

# --- ETAPA DE CONFIGURAÇÃO ---
# 1. Crie uma conta no Hugging Face: https://huggingface.co/
# 2. Gere um Token de Acesso em Settings > Access Tokens
# 3. No Colab, execute a célula de login e cole seu token.
# login() # Descomente esta linha para fazer o login no Colab

# --- Constantes de Configuração ---
# Use o modelo 'base', que se provou melhor para nosso dataset
MODEL_NAME = "neuralmind/bert-base-portuguese-cased" 
KAGGLE_DATASET_PATH = "youtube_comment_sentiment.csv"
LOCAL_SAVE_PATH = "./meu_modelo_sentimento_youtube_local"
# Defina o nome do seu repositório no Hugging Face Hub
# Substitua "seu-usuario-hf" pelo seu nome de usuário
HUB_MODEL_ID = "seu-usuario-hf/sentinela-tube-sentiment-model" 

def run_training_phase():
    """
    Executa todo o fluxo de treinamento, salva localmente e envia para o Hub.
    """
    print("==================================================")
    print("🚀 INICIANDO TREINAMENTO DO MODELO")
    print("==================================================")

    # Carregar e preparar o dataset
    try:
        df = pd.read_csv(KAGGLE_DATASET_PATH, encoding='utf-8')
    except Exception:
        df = pd.read_csv(KAGGLE_DATASET_PATH, encoding='latin-1')

    df = df[['Comment', 'Sentiment']].dropna()
    df.columns = ['text', 'label_text']
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['label_text'].map(label_map)
    print("✅ Dataset carregado e pré-processado.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # Carregar Tokenizador e Modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    print("✅ Modelo e tokenizador base carregados.")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    class YouTubeCommentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = YouTubeCommentDataset(train_encodings, train_labels)
    test_dataset = YouTubeCommentDataset(test_encodings, test_labels)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    # Configurações de Treinamento
    training_args = TrainingArguments(
        output_dir='./results',
        report_to="none",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        logging_steps=500,
        push_to_hub=True, # Habilita o envio para o Hub
        hub_model_id=HUB_MODEL_ID, # Define o nome do repositório no Hub
        hub_strategy="end" # Envia o modelo final ao término do treino
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    
    print("\nIniciando o processo de fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning concluído!")

    # O `trainer` com `push_to_hub=True` já envia o modelo e o tokenizador.
    print(f"✅ Modelo enviado com sucesso para o Hub: https://huggingface.co/{HUB_MODEL_ID}")
    
    return True

if __name__ == "__main__":
    run_training_phase()