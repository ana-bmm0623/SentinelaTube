# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_api import extrair_id_video, buscar_comentarios
from analysis_tools import carregar_modelo_sentimento, carregar_modelo_emocao, prever_analise, extrair_topicos

@st.cache_resource
def carregar_modelos():
    st.write("Carregando modelos de IA... (Isso acontece apenas na primeira execução)")
    tokenizer_sent, model_sent = carregar_modelo_sentimento()
    tokenizer_emo, model_emo = carregar_modelo_emocao()
    return tokenizer_sent, model_sent, tokenizer_emo, model_emo

st.set_page_config(page_title="SentinelaTube", layout="wide")
st.title("🤖 SentinelaTube: Análise Profunda de Comentários")

tok_sent, mod_sent, tok_emo, mod_emo = carregar_modelos()

if not mod_sent or not mod_emo:
    st.error("Modelos não encontrados. Certifique-se de que as pastas 'meu_modelo_sentimento' e 'meu_modelo_emocoes' estão no diretório correto.")
else:
    st.sidebar.header("Configurações da Análise")
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
        st.sidebar.success("Chave da API carregada!")
    except KeyError:
        st.sidebar.error("Adicione sua chave da API no arquivo .streamlit/secrets.toml")
        st.stop()

    video_url = st.sidebar.text_input("Link do Vídeo do YouTube:")
    max_comments = st.sidebar.slider("Nº Máximo de Comentários:", 50, 1000, 200, 50)

    if st.sidebar.button("Analisar"):
        if video_url and api_key:
            video_id = extrair_id_video(video_url)
            if video_id:
                with st.spinner(f"Buscando e analisando até {max_comments} comentários..."):
                    try:
                        comentarios = buscar_comentarios(video_id, api_key, max_comments)
                        if comentarios:
                            sentimentos = prever_analise(comentarios, tok_sent, mod_sent, {0:'Negativo', 1:'Neutro', 2:'Positivo'})
                            emocoes = prever_analise(comentarios, tok_emo, mod_emo, {0:'Alegria', 1:'Medo', 2:'Raiva', 3:'Nojo', 4:'Surpresa', 5:'Tristeza'})
                            topicos = extrair_topicos(comentarios)
                            results_df = pd.DataFrame({'Comentário': comentarios, 'Sentimento': sentimentos, 'Emoção': emocoes})

                            st.success("Análise Concluída!")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.subheader("Resumo de Sentimentos")
                                st.bar_chart(results_df['Sentimento'].value_counts())
                            with c2:
                                st.subheader("Resumo de Emoções")
                                st.bar_chart(results_df['Emoção'].value_counts())

                            st.subheader("Nuvem de Tópicos Discutidos")
                            if len(topicos) > 0:
                                wordcloud = WordCloud(width=800, height=250, background_color='white').generate(" ".join(topicos))
                                st.image(wordcloud.to_array())
                            
                            st.subheader("Análise Detalhada")
                            st.dataframe(results_df)
                    except Exception as e:
                        st.error(f"Ocorreu um erro: {e}")
            else:
                st.error("URL do vídeo inválida.")
        else:
            st.warning("Por favor, insira a URL do vídeo.")