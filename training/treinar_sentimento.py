# Célula de Treinamento de Sentimento (VERSÃO EM INGLÊS)
# -*- coding: utf-8 -*-
import torch
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- Constantes (AJUSTADAS PARA INGLÊS) ---
# MUDANÇA 1: Usando um modelo base pré-treinado em inglês
MODEL_NAME = "distilbert-base-uncased" 
# Usando o mesmo dataset que contém exemplos em inglês
DATASET_PATH = "youtube_comment_sentiment.csv" 
# MUDANÇA 2: Salvando o novo modelo em uma pasta diferente
SAVED_MODEL_PATH = "./sentiment_model_en" 

def run_sentiment_training_en():
    print("🚀 INICIANDO TREINAMENTO DO MODELO DE SENTIMENTO (INGLÊS)")
    
    # Carregar e preparar o dataset
    try:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    except Exception:
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')

    # A estrutura do CSV é a mesma
    df = df[['Comment', 'Sentiment']].dropna()
    df.columns = ['text', 'label_text']
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['label_text'].map(label_map)
    print(f"✅ Dataset com {len(df)} exemplos carregado.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # Carregar o novo Modelo e Tokenizador em inglês
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels): self.encodings, self.labels = encodings, labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]); return item
        def __len__(self): return len(self.labels)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    def compute_metrics(pred): return {'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}

    # Argumentos de Treinamento
    training_args = TrainingArguments(
        output_dir='./results_sentiment_en',
        report_to="none",
        num_train_epochs=3,
        eval_strategy="epoch",
        logging_steps=500,
        per_device_train_batch_size=32, # Aumentado para o DistilBERT que é mais leve
        learning_rate=2e-5,
        load_best_model_at_end=True,
        save_strategy="epoch"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    
    # Inicia o treinamento
    trainer.train()
    
    print("✅ Fine-tuning de Sentimento (Inglês) concluído!")
    
    # Salva e compacta o modelo final
    trainer.save_model(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    print(f"✅ Modelo de Sentimento em Inglês salvo em '{SAVED_MODEL_PATH}'")

    print(f"🗜️ Compactando o modelo em '{SAVED_MODEL_PATH}.zip'...")
    shutil.make_archive(SAVED_MODEL_PATH, 'zip', SAVED_MODEL_PATH)
    print("✅ Modelo compactado com sucesso!")
    
    return True

if __name__ == "__main__":
    import shutil
    run_sentiment_training_en()