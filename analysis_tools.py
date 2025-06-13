# analysis_tools.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from config import *

def carregar_modelo(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except OSError:
        return None, None

@torch.no_grad()
def prever_analise(textos: list, tokenizer, model, label_map):
    inputs = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=128)
    logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=1)
    return [label_map.get(p.item(), "unknown") for p in preds]

def extrair_topicos(comentarios: list, top_n: int = 10):
    try:
        # ATUALIZADO PARA INGLÊS
        stop_words = stopwords.words('english')
        stop_words.extend(['http', 'https', 'like', 'video', 'youtube', 'com', 'im', 'ive'])
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, max_features=top_n)
        vectorizer.fit_transform(comentarios)
        return vectorizer.get_feature_names_out()
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []