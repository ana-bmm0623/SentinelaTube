# config.py
# Caminhos para as pastas dos modelos que você descompactou
SAVED_MODEL_PATH_SENTIMENT = "./sentiment_model_en"
SAVED_MODEL_PATH_EMOTION = "./emotion_model_en"

# Mapeamento para o Modelo de Sentimento (3 classes)
LABEL_MAP_SENTIMENT = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Mapeamento para o Modelo de Emoções (28 classes do GoEmotions)
LABEL_MAP_EMOTION = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 
    15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 
    25: 'sadness', 26: 'surprise', 27: 'neutral'
}