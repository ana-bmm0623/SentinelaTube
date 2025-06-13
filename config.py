# config.py
# Este arquivo centraliza todas as configurações e constantes do projeto SentinelaTube.

# --- CAMINHOS PARA OS MODELOS TREINADOS ---
# Caminho para a pasta onde o modelo de sentimento (Positivo/Negativo/Neutro) foi salvo.
SAVED_MODEL_PATH_SENTIMENT = "./meu_modelo_sentimento"

# Caminho para a pasta onde o modelo de emoções será salvo.
SAVED_MODEL_PATH_EMOTION = "./meu_modelo_emocoes"


# --- MAPEAMENTO DE RÓTULOS (LABELS) ---

# Mapeamento para o Modelo de Sentimento (3 classes)
LABEL_MAP_SENTIMENT = {
    0: 'Negativo', 
    1: 'Neutro', 
    2: 'Positivo'
}

# Mapeamento para o Modelo de Emoções (28 classes)
# Baseado no dataset GoEmotions (usado no script treinar_emocoes.py)
LABEL_MAP_EMOTION = {
    0: 'admiração',
    1: 'diversão',
    2: 'raiva',
    3: 'aborrecimento',
    4: 'cuidado',
    5: 'confusão',
    6: 'curiosidade',
    7: 'desejo',
    8: 'decepção',
    9: 'desaprovação',
    10: 'nojo',
    11: 'constrangimento',
    12: 'entusiasmo',
    13: 'medo',
    14: 'gratidão',
    15: 'tristeza',
    16: 'alegria',
    17: 'amor',
    18: 'nervosismo',
    19: 'otimismo',
    20: 'orgulho',
    21: 'percepção',
    22: 'alívio',
    23: 'remorso',
    24: 'tristeza', # 'sadness' aparece novamente, pode ser um sinônimo no dataset
    25: 'surpresa',
    26: 'neutro',   # O dataset de emoções também inclui uma categoria neutra
    27: 'aprovação' # 'approval' está duplicado, mas mantemos para consistência com o dataset original
}