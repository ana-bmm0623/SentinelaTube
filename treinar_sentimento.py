# treinar_sentimento.py
import torch
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
DATASET_PATH = "youtube_comment_sentiment.csv"
SAVED_MODEL_PATH = "./meu_modelo_sentimento"

def run_sentiment_training():
    print("🚀 INICIANDO TREINAMENTO DO MODELO DE SENTIMENTO")
    
    try:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    except Exception:
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')

    df = df[['Comment', 'Sentiment']].dropna()
    df.columns = ['text', 'label_text']
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['label_text'].map(label_map)
    print("✅ Dataset de Sentimento carregado.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

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

    training_args = TrainingArguments(output_dir='./results_sentiment', report_to="none", num_train_epochs=3, evaluation_strategy="epoch", logging_steps=500, per_device_train_batch_size=16)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    
    trainer.train()
    print("✅ Fine-tuning de Sentimento concluído!")

    trainer.save_model(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    print(f"✅ Modelo de Sentimento salvo em '{SAVED_MODEL_PATH}'")

    print(f"🗜️ Compactando o modelo em '{SAVED_MODEL_PATH}.zip'...")
    shutil.make_archive(SAVED_MODEL_PATH, 'zip', SAVED_MODEL_PATH)
    print("✅ Modelo compactado com sucesso!")
    
    return True

if __name__ == "__main__":
    run_sentiment_training()