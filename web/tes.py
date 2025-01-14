

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk memuat model dan data
def load_data(file):
    data = pd.read_csv(file, sep=';')  # Memuat CSV dengan pemisah koma
    return data

# Fungsi untuk pelatihan dan prediksi
def classify_sentiment(texts, labels):
    # Menggunakan TF-IDF untuk representasi teks
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Memisahkan data untuk pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4, random_state=42)
    
    # Membuat model SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    # Memprediksi hasil
    y_pred = model.predict(X_test)
    
    # Menghitung hasil evaluasi
    acc = accuracy_score(y_test, y_pred)  # Akurasi
    conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion Matrix
    report = classification_report(y_test, y_pred, output_dict=True)  # Classification report
    
    return acc, conf_matrix, report, y_pred, y_test, X_test

st.title('Klasifikasi Sentimen')

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data yang diunggah:", df)
    
 
    if 'text' in df.columns and 'label' in df.columns:
        # Melakukan klasifikasi
        acc, conf_matrix, report, y_pred, y_test, X_test = classify_sentiment(df['text'], df['label'])
        
        # Menampilkan hasil akurasi
        st.write(f"Akurasi Model: {acc * 100:.2f}%")
        
        # Menampilkan classification report
        st.write("Hasil Klasifikasi Sentimen:")
        st.text(classification_report(y_test, y_pred))
        
        # Menampilkan Confusion Matrix
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Positif', 'Negatif', 'Netral'], yticklabels=['Positif', 'Negatif', 'Netral'])
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        st.pyplot(fig)
        
        # Memastikan ukuran array cocok dengan panjang indeks X_test
        test_indices = range(len(X_test.toarray()))  # Menggunakan panjang X_test untuk mendapatkan indeks yang sesuai
        
        # Membuat DataFrame dengan teks yang sesuai
        df_test = pd.DataFrame({
            'Text': df.iloc[test_indices]['text'].values,  
            'Label Asli': y_test,
            'Prediksi Sentimen': y_pred
        })
        
        st.write("Klasifikasi Sentimen (Label Asli vs Prediksi Sentimen):")
        st.dataframe(df_test)
        
    else:
        st.error("CSV harus memiliki kolom 'text' dan 'label'.")
