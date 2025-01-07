import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image


@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')  
    return data

data = load_data()

st.write("Nama kolom dalam data:", data.columns.tolist())

tweet_column = 'tweet_preprocessed'  
sentiment_column = 'sentiment'  


st.sidebar.title("Menu")
option = st.sidebar.radio("Pilih Menu", ('Show Wordcloud', 'Show Grafik Sentimen'))


if option == 'Show Wordcloud':
    sentiment_option = st.sidebar.radio("Pilih Opsi Wordcloud", ('Semua', 'Positif', 'Negatif', 'Netral'))  
    if sentiment_option == 'Semua':
        filtered_data = data
    else:
        filtered_data = data[data[sentiment_column] == sentiment_option]

    text = " ".join(tweet for tweet in filtered_data[tweet_column])


    mask_image = Image.open('map.webp')  
    mask_array = np.array(mask_image)


    wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask_array, contour_color='black', contour_width=1).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


elif option == 'Show Grafik Sentimen':
    chart_type = st.sidebar.radio("Pilih Tipe Grafik", ('Histogram', 'Pie Chart'))
    sentiment_counts = data[sentiment_column].value_counts()

    if chart_type == 'Histogram':
        st.bar_chart(sentiment_counts)

    elif chart_type == 'Pie Chart':
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  
        st.pyplot(fig)
