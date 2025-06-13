# analysis_tools.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from config import SAVED_MODEL_PATH_SENTIMENT, SAVED_MODEL_PATH_EMOTION, LABEL_MAP_SENTIMENT, LABEL_MAP_EMOTION

def carregar_modelo_sentimento():
    try: return AutoTokenizer.from_pretrained(SAVED_MODEL_PATH_SENTIMENT), AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH_SENTIMENT)
    except OSError: return None, None

def carregar_modelo_emocao():
    try: return AutoTokenizer.from_pretrained(SAVED_MODEL_PATH_EMOTION), AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH_EMOTION)
    except OSError: return None, None

@torch.no_grad()
def prever_analise(textos: list, tokenizer, model, label_map):
    inputs = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=128)
    logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=1)
    return [label_map[p.item()] for p in preds]

def extrair_topicos(comentarios: list, top_n: int = 10) -> list:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = stopwords.words('portuguese')
    stop_words.extend(['pra', 'q', 'vc', 'tá'])
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, max_features=top_n)
    vectorizer.fit_transform(comentarios)
    return vectorizer.get_feature_names_out()