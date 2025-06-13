# Célula de Treinamento de Emoções - Lendo Arquivos .txt Locais
# -*- coding: utf-8 -*-
import torch
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- Constantes ---
MODEL_NAME = "distilbert-base-uncased"
SAVED_MODEL_PATH = "./emotion_model_en"
# Lista dos arquivos de dados que você enviou
DATA_FILES = ['train.txt', 'test.txt', 'val.txt']

def run_emotion_training():
    print("🚀 INICIANDO TREINAMENTO DO MODELO DE EMOÇÕES")
    
    # --- LÓGICA DEFINITIVA PARA CARREGAR OS ARQUIVOS .TXT LOCAIS ---
    print("\n[Fase Emoção] Carregando e processando arquivos .txt locais...")
    try:
        all_data = []
        # Itera sobre cada um dos seus arquivos .txt
        for file_name in DATA_FILES:
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    # Divide a linha no caractere ';' para separar texto e rótulo
                    parts = line.strip().split(';')
                    if len(parts) == 2:
                        text, label = parts
                        all_data.append({'text': text, 'label_text': label})
        
        # Cria um único DataFrame com todos os dados
        df = pd.DataFrame(all_data)
        
    except FileNotFoundError as e:
        print(f"❌ ERRO: Arquivo não encontrado - {e}.")
        print("Certifique-se de que os 3 arquivos .txt (train.txt, test.txt, val.txt) foram enviados para o Colab.")
        return False
    except Exception as e:
        print(f"❌ ERRO ao carregar ou processar o dataset de emoções: {e}")
        return False
    
    # Mapeia as emoções de texto para números
    labels_map = {label: i for i, label in enumerate(df['label_text'].unique())}
    df['label'] = df['label_text'].map(labels_map)
    num_labels = len(labels_map)
    
    print(f"✅ Dataset com {len(df)} exemplos e {num_labels} classes de emoção carregado.")
    print("Mapeamento de emoções criado:", labels_map)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # O resto do processo é o mesmo que já validamos...
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    class EmotionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels): self.encodings, self.labels = encodings, labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]); return item
        def __len__(self): return len(self.labels)

    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    def compute_metrics(pred): return {'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}

    training_args = TrainingArguments(
        output_dir='./results_emocoes_en',
        report_to="none",
        num_train_epochs=3,
        eval_strategy="epoch",
        logging_steps=500,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        save_strategy="epoch"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    
    trainer.train()
    print("✅ Fine-tuning de Emoções concluído!")
    
    trainer.save_model(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    print(f"✅ Modelo de Emoções salvo em '{SAVED_MODEL_PATH}'")

    print(f"🗜️ Compactando o modelo em '{SAVED_MODEL_PATH}.zip'...")
    shutil.make_archive(SAVED_MODEL_PATH, 'zip', SAVED_MODEL_PATH)
    print("✅ Modelo compactado com sucesso!")
    
    return True

if __name__ == "__main__":
    import shutil
    run_emotion_training()