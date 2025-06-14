# config.py - VERSÃO ATUALIZADA FINAL
SAVED_MODEL_PATH_SENTIMENT = "./sentiment_model_en"
SAVED_MODEL_PATH_EMOTION = "./emotion_model_en" # Pasta do novo modelo

LABEL_MAP_SENTIMENT = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Mapeamento para o novo Modelo de Emoções (6 classes)
LABEL_MAP_EMOTION = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}