import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Analisis Sentimen Komentar Publik")
st.markdown("**Metode: Support Vector Machine (SVM)**")
st.markdown("Aplikasi ini menganalisis sentimen komentar publik seperti positif, negatif, dan netral menggunakan metode SVM.")

# Upload File CSV
uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset yang diunggah:")
    st.dataframe(df.head())

    # Pisahkan kolom 
    if "text;label" in df.columns:
        df[['text', 'label']] = df['text;label'].str.split(';', expand=True)
        df.drop(columns=['text;label'], inplace=True)  # Hapus kolom lama


    st.write("Dataset setelah pemisahan kolom:")
    st.dataframe(df.head())
   
    required_column = 'text'
    if required_column in df.columns:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['text'])

        # Pastikan ada kolom sentimen
        if 'label' in df.columns:
            y = df['label'] 

            # Split Data 80 20
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training Model
            st.subheader("Training Model")
            kernel = st.selectbox("Pilih Kernel SVM", ["linear", "rbf", "poly", "sigmoid"])
            model = SVC(kernel=kernel)
            model.fit(X_train, y_train)

            # Evaluasi Model
            y_pred = model.predict(X_test)
            st.write("Hasil Evaluasi Model:")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Prediksi Real-time
            st.subheader("Prediksi Sentimen Real-time")
            user_input = st.text_input("Masukkan komentar untuk dianalisis:")
            if user_input:
                user_input_vec = vectorizer.transform([user_input])
                pred = model.predict(user_input_vec)
                st.write(f"Prediksi Sentimen: {pred[0]}")
        else:
            st.error("Kolom 'sentimen' tidak ditemukan dalam dataset.")
    else:
        st.error(f"Kolom '{required_column}' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom ini.")
else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")