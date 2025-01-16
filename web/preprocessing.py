import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from nltk.data import find
from nltk import download

# Download resource yang diperlukan dari nltk
# nltk.download('punkt')  
# nltk.download('stopwords')
# nltk.download('punkt_tab')

def show_prepocessing():
    def download_if_needed(resource):
        try:
            find(f'tokenizers/{resource}')
            print(f"Resource '{resource}' sudah ada.")
        except LookupError:
            print(f"Resource '{resource}' tidak ditemukan. Mendownload...")
            download(resource)
    # Mengunduh resource jika belum ada
            download_if_needed('punkt')
            download_if_needed('punkt_tab')
            download_if_needed('stopwords')
    # Fitur upload file
    st.title('Upload dan Proses Data CSV')

    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        # Membaca file yang diunggah
        df = pd.read_csv(uploaded_file, delimiter=';')
        
        # Menampilkan jumlah data yang diunggah
        st.write(f"Total Data Awal: {len(df)} baris")
        st.dataframe(df)  # Menampilkan seluruh data yang diunggah

        # Penghapusan kolom yang tidak diperlukan
        df = df.drop(columns=[
            'conversation_id_str',
            'created_at',
            'favorite_count',
            'id_str',
            'image_url',
            'in_reply_to_screen_name',
            'lang',
            'location',
            'quote_count',
            'reply_count',
            'retweet_count',
            'tweet_url',
            'user_id_str'
        ])

        # Menampilkan data setelah penghapusan kolom
        st.write("Data setelah penghapusan kolom:")
        st.dataframe(df)

        # Menghapus duplikat dan missing values
        df = df.drop_duplicates()
        df = df.dropna()

        # Menampilkan data setelah penghapusan duplikat dan missing values
        st.write("Data setelah menghapus duplikat dan missing values:")
        st.dataframe(df)

        # Pembersihan Teks
        df['tweet_lower'] = df['full_text'].str.lower()
        df['tweet_no_url'] = df['tweet_lower'].apply(lambda x: re.sub(r'http\S+', '', x))
        df['tweet_no_mention_hashtag'] = df['tweet_no_url'].apply(lambda x: re.sub(r'@\w+|#\w+', '', x))
        df['tweet_no_punctuation'] = df['tweet_no_mention_hashtag'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df['tweet_no_numbers'] = df['tweet_no_punctuation'].apply(lambda x: re.sub(r'\d+', '', x))
        df['tweet_clean'] = df['tweet_no_numbers'].apply(lambda x: x.strip())

        # Menampilkan data setelah pembersihan teks
        st.write("Data setelah pembersihan teks:")
        st.dataframe(df)

        # Tokenisasi
        df['tweet_tokenized'] = df['tweet_clean'].apply(word_tokenize)

        # Menampilkan data setelah tokenisasi
        st.write("Data setelah tokenisasi:")
        st.dataframe(df)

        # Gabungkan stopword bawaan dengan stopword eksternal
        external_stopwords = [
            'btw', 'gpp', 'gmna', 'gmn', 'gt', 'tbh', 'gw', 'gua', 'lu', 'elo', 'lo', 'sm', 'jd', 'dgn', 'tdk',
            'yg', 'dr', 'spt', 'utk', 'sbg', 'pd', 'msh', 'blm', 'udh', 'ud', 'dl', 'krn', 'klo', 'kl', 'knp', 'hrs',
            'bs', 'tp', 'aj', 'lg', 'ktnya', 'lbh', 'skrg', 'tmn', 'tmen','pnya', 'mshh', 'km', 'sy', 'q', 'ok',
            'okk', 'pke', 'pk', 'sdh', 'dmn', 'td', 'sd', 'sb', 'cma', 'nggk', 'ngga', 'nda', 'dg', 'bnr', 'brp',
            'py', 'bgmn', 'smoga', 'tpnya', 'bgitu', 'aplh', 'bsk', 'mslh', 'trs', 'syg', 'kyknya', 'gmana', 'kmu',
            'ajh', 'mau', 'kmn', 'apa', 'blg', 'akh', 'trkdang', 'sndri', 'ank', 'org', 'trus', 'mrka', 'aja', 'bgt',
            'dn', 'mksd', 'ttg', 'dlm', 'ttp', 'rt', 'rw', 'asik', 'brb', 'bro', 'bt', 'cmiww', 'cmiiw', 'cpt', 'dpt',
            'dri', 'dtd', 'eaa', 'ganteng', 'ikut', 'jgn', 'kapal', 'klr', 'kti', 'lgsg', 'maap', 'mlm', 'mrk', 'nnti',
            'nyoba', 'pls', 'prnh', 'qt', 'rl', 'sep', 'sbnr', 'tdr', 'trs', 'ttg', 'wkwk', 'ya', 'z', 'zz', 'aq'
        ]
        combined_stopwords = stopwords.words('indonesian') + external_stopwords
        stop_factory_added = StopWordRemoverFactory()
        stop_factory_added.stop_words = combined_stopwords
        stopword_remover = stop_factory_added.create_stop_word_remover()

        df['tweet_no_stopwords'] = df['tweet_tokenized'].apply(
            lambda x: [stopword_remover.remove(word) for word in x if word not in combined_stopwords]
        )

        # Menampilkan data setelah menghapus stopwords
        st.write("Data setelah menghapus stopwords:")
        st.dataframe(df)

        # Stemming
        stem_factory = StemmerFactory()
        stemmer = stem_factory.create_stemmer()
        df['tweet_stemmed'] = df['tweet_no_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])

        # Menampilkan data setelah stemming
        st.write("Data setelah stemming:")
        st.dataframe(df)

        # Gabungkan kata-kata yang telah diproses
        df['tweet_preprocessed'] = df['tweet_stemmed'].apply(lambda x: ' '.join(x))

        # Menampilkan data setelah pre-processing
        st.write("Data setelah pre-processing:")
        st.dataframe(df)

        # Menambahkan kolom label
        df['label'] = df['tweet_preprocessed'].apply(lambda x: 1 if 'positif' in x else 0)  # Contoh label berdasarkan kata kunci

        # Menampilkan data setelah penambahan kolom label
        st.write("Data setelah penambahan kolom label:")
        st.dataframe(df)

        # Menyimpan data yang telah diproses ke file CSV
        df.to_csv('temp/data_preprocessing.csv', index=False)
        st.write("Data telah disimpan ke data_preprocessing.csv")

        # Menyimpan data akhir
        df_final = df.drop(columns=[
            'full_text',
            'username',
            'tweet_lower',
            'tweet_no_url',
            'tweet_no_mention_hashtag',
            'tweet_no_punctuation',
            'tweet_no_numbers',
            'tweet_clean',
            'tweet_tokenized',
            'tweet_no_stopwords',
            'tweet_stemmed'
        ], errors='ignore')

        # Menampilkan data akhir
        st.write("Data setelah penghapusan kolom akhir:")
        st.dataframe(df_final)

        # Menyimpan file CSV akhir
        df_final.to_csv('temp/data.csv', index=False)
        st.write("Data akhir telah disimpan ke data.csv")
