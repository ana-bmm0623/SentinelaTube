# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_api import extrair_id_video, buscar_comentarios
from analysis_tools import carregar_modelo, prever_analise, extrair_topicos
from config import *

@st.cache_resource
def carregar_modelos_cached():
    st.write("Loading AI models... (This happens only on the first run)")
    tok_sent, mod_sent = carregar_modelo(SAVED_MODEL_PATH_SENTIMENT)
    tok_emo, mod_emo = carregar_modelo(SAVED_MODEL_PATH_EMOTION)
    return tok_sent, mod_sent, tok_emo, mod_emo

st.set_page_config(page_title="InsightTube", layout="wide")
st.title("💡 InsightTube: Deep YouTube Comment Analyzer")
st.write("This app uses two specialized Transformer models to perform a deep analysis of sentiment, emotions, and topics from YouTube comments.")

tok_sent, mod_sent, tok_emo, mod_emo = carregar_modelos_cached()

if not mod_sent or not mod_emo:
    st.error("One or more trained models were not found. Make sure the 'sentiment_model_en' and 'emotion_model_en' folders are in the project directory.")
else:
    st.sidebar.header("Analysis Configuration")
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
        st.sidebar.success("YouTube API Key loaded successfully!")
    except KeyError:
        st.sidebar.error("Add your YouTube API Key to the .streamlit/secrets.toml file.")
        st.stop()

    video_url = st.sidebar.text_input("YouTube Video Link:")
    max_comments = st.sidebar.slider("Max Comments to Analyze:", 50, 1000, 200, 50)

    if st.sidebar.button("Analyze Comments"):
        if video_url and api_key:
            video_id = extrair_id_video(video_url)
            if video_id:
                with st.spinner(f"Fetching and analyzing up to {max_comments} comments..."):
                    try:
                        comentarios = buscar_comentarios(video_id, api_key, max_comments)
                        if comentarios:
                            sentimentos = prever_analise(comentarios, tok_sent, mod_sent, LABEL_MAP_SENTIMENT)
                            emocoes = prever_analise(comentarios, tok_emo, mod_emo, LABEL_MAP_EMOTION)
                            topicos = extrair_topicos(comentarios)
                            results_df = pd.DataFrame({'Comment': comentarios, 'Sentiment': sentimentos, 'Emotion': emocoes})

                            st.success("Analysis Complete!")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Sentiment Summary")
                                st.bar_chart(results_df['Sentiment'].value_counts())
                            with col2:
                                st.subheader("Emotion Summary")
                                st.bar_chart(results_df['Emotion'].value_counts())

                            st.subheader("Main Topics Discussed")
                            if len(topicos) > 0:
                                wordcloud = WordCloud(width=800, height=250, background_color='white', colormap='viridis').generate(" ".join(topicos))
                                st.image(wordcloud.to_array())
                            
                            st.subheader("Detailed Analysis")
                            st.dataframe(results_df)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Invalid YouTube URL.")
        else:
            st.warning("Please provide a video URL.")