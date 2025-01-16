import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from home import show_home  
from svm import show_svm  
from preprocessing import show_prepocessing  
from streamlit_option_menu import option_menu

@st.cache_data
def load_data():
    data = pd.read_csv('data/data.csv', sep=';')  
    return data

data = load_data()

tweet_column = 'tweet_preprocessed'
sentiment_column = 'label'

with st.sidebar:
    col1, col2 = st.columns([1, 4]) 
    with col1:
        try:
            logo = Image.open('image/logo.png')  
            st.image(logo, use_column_width=True)  
        except FileNotFoundError:
            st.warning("Logo not found!")
    with col2:
        st.markdown(
            """
            <div style='display: flex; align-items: center; height: 100%; margin: 0; padding-top: 5px;'>
                <h1 style='font-size: 20px; margin: 0;'>APES</h1>
            </div>
            """,
            unsafe_allow_html=True
        )  

 
    option = option_menu(
        menu_title=None, 
        options=["Home","Preprocessing Data", "Show Wordcloud", "Show Grafik Sentimen", "Clasification SVM"], 
        icons=["house", "sliders", "cloud-sun", "bar-chart", "search"],  
        default_index=0, 
        orientation="vertical", 
    )


if option == 'Home':
    show_home() 

elif option == 'Preprocessing Data':
    show_prepocessing()  

elif option == 'Show Wordcloud':
 
    sentiment_option = st.selectbox(
        "Pilih Opsi Sentimen untuk Wordcloud", 
        ['Semua', 'Positif', 'Negatif', 'Netral'], 
        index=0 
    )

    if sentiment_option == 'Semua':
        filtered_data = data
    else:
        filtered_data = data[data[sentiment_column] == sentiment_option]

    st.markdown("### Wordcloud Berdasarkan Sentimen")
    st.markdown(
        "Wordcloud ini menunjukkan kata-kata yang paling sering muncul dalam teks berdasarkan kategori sentimen yang dipilih."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    text = " ".join(tweet for tweet in filtered_data[tweet_column])

    try:
        mask_image = Image.open('image/map.webp') 
        mask_array = np.array(mask_image)
    except FileNotFoundError:
        mask_array = None 

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        mask=mask_array,
        contour_color='black',
        contour_width=1
    ).generate(text)

    st.image(wordcloud.to_array(), use_column_width=True)
    
    st.subheader("Data yang Digunakan untuk Wordcloud")
    st.dataframe(filtered_data)


elif option == 'Show Grafik Sentimen':
    st.markdown("### Grafik Sentimen")
    st.markdown(
        "Grafik ini menunjukkan distribusi data sentimen berdasarkan jumlah dan persentasenya."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    sentiment_counts = data[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    total = sentiment_counts['Count'].sum()
    sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total * 100).round(2)

    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        text=sentiment_counts['Percentage'].apply(lambda x: f"{x}%"),  
        color_discrete_map={
            'Positif': '#4CAF50',
            'Negatif': '#F44336',
            'Netral': '#FFC107'
        },
        title='Sentiment Analysis',
        labels={'Sentiment': 'Sentiment', 'Count': 'Jumlah'},
        template='plotly_white'
    )

    fig.update_traces(
        textposition='outside', 
        textfont_size=12
    )
    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, family='Arial'),
        xaxis=dict(title='Sentiment', tickangle=-45),
        yaxis=dict(title='Jumlah', gridcolor='rgba(200,200,200,0.3)'),
        showlegend=False
    )

    st.plotly_chart(fig)
    st.subheader("Seluruh Data yang Digunakan untuk Grafik")
    st.dataframe(data) 


elif option == 'Clasification SVM':
    show_svm()  
