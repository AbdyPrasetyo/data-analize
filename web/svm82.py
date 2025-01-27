import streamlit as st 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from imblearn.over_sampling import SMOTE
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json

def show_svm82():
    # Fungsi untuk memuat model dan data
    def load_data(file):
        data = pd.read_csv(file, sep=';')  # Memuat CSV dengan pemisah koma
        return data

    # Fungsi untuk preprocessing teks
    def preprocess_text(text):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('indonesian'))  # Sesuaikan dengan bahasa
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)

    # Fungsi untuk pelatihan dan prediksi
    def classify_sentiment(texts, labels):
        # Menggunakan TF-IDF untuk representasi teks
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        
        # Menyeimbangkan data dengan SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, labels)

        # Pembagian data dengan stratifikasi
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Hyperparameter tuning dengan GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Menggunakan model terbaik dari GridSearchCV
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Menghitung hasil evaluasi
        acc = accuracy_score(y_test, y_pred)  # Akurasi
        conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion Matrix
        report = classification_report(y_test, y_pred, output_dict=True)  # Classification report
        
        return acc, conf_matrix, report, y_pred, y_test, X_test, X_train

    st.title('Klasifikasi Sentimen')

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.title("Data yang diunggah:")
        st.write(df)
        
        if 'tweet_preprocessed' in df.columns and 'label' in df.columns:
            # Preprocessing teks
            df['tweet_preprocessed'] = df['tweet_preprocessed'].apply(preprocess_text)

            # Melakukan klasifikasi
            acc, conf_matrix, report, y_pred, y_test, X_test, X_train = classify_sentiment(df['tweet_preprocessed'], df['label'])
            
            # Menampilkan distribusi panjang teks
            st.title("Distribusi Panjang Teks pada Dataset Pelatihan dan Pengujian:")
            lengths_train = df['tweet_preprocessed'][:X_train.shape[0]].str.len()
            lengths_test = df['tweet_preprocessed'][-X_test.shape[0]:].str.len()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.hist(lengths_train, bins=50, label="Train Tweets", color="darkblue", alpha=0.7)
            plt.hist(lengths_test, bins=50, label="Test Tweets", color="skyblue", alpha=0.7)
            plt.xlabel("Panjang Teks")
            plt.ylabel("Frekuensi")
            plt.title("Distribusi Panjang Teks")
            plt.legend()
            st.pyplot(fig)
            st.title(f"Akurasi Model: {acc * 100:.2f}%")
            # Menampilkan hasil akurasi
            st.title("Classification Report")
            st.write("Classification Report adalah  laporan yang digunakan untuk mengevaluasi performa model klasifikasi dalam pembelajaran mesin. Laporan ini memberikan metrik evaluasi untuk setiap kelas dalam dataset, seperti precision, recall, F1-score, dan support.")
            result_table = pd.DataFrame({
                'Label': ['Negatif', 'Netral', 'Positif', 'accuracy', 'macro avg', 'weighted avg'],
                'precision': [
                    f"{report['Negatif']['precision']:.2f}",
                    f"{report['Netral']['precision']:.2f}",
                    f"{report['Positif']['precision']:.2f}",
                    f"{acc:.2f}",
                    f"{report['macro avg']['precision']:.2f}",
                    f"{report['weighted avg']['precision']:.2f}"
                ],
                'recall': [
                    f"{report['Negatif']['recall']:.2f}",
                    f"{report['Netral']['recall']:.2f}",
                    f"{report['Positif']['recall']:.2f}",
                    f"{acc:.2f}",
                    f"{report['macro avg']['recall']:.2f}",
                    f"{report['weighted avg']['recall']:.2f}"
                ],
                'f1-score': [
                    f"{report['Negatif']['f1-score']:.2f}",
                    f"{report['Netral']['f1-score']:.2f}",
                    f"{report['Positif']['f1-score']:.2f}",
                    f"{acc:.2f}",
                    f"{report['macro avg']['f1-score']:.2f}",
                    f"{report['weighted avg']['f1-score']:.2f}"
                ],
                'support': [
                    int(report['Negatif']['support']),
                    int(report['Netral']['support']),
                    int(report['Positif']['support']),
                    len(y_test),
                    len(y_test),
                    len(y_test)
                ]
            })
            
            st.dataframe(result_table, use_container_width=True, hide_index=True)

            
            # Menampilkan Confusion Matrix
            st.title("Confusion Matrix:")
            st.write("Confusion Matrix adalah tabel yang digunakan untuk mengevaluasi performa model klasifikasi SVM dengan membandingkan hasil prediksi model terhadap data aktual (label sebenarnya).")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Positif', 'Negatif', 'Netral'], yticklabels=['Positif', 'Negatif', 'Netral'])
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            st.pyplot(fig)

            pred_counts = pd.Series(y_pred).value_counts()  # Gunakan y_pred yang sesuai dengan data uji
            labels = pred_counts.index
            sizes = pred_counts.values
            total_data = sizes.sum()

            fig2 = px.pie(values=pred_counts, names=labels, hole=0.7, 
                        color_discrete_sequence=px.colors.qualitative.Pastel, 
                        title="Prediksi SVM")

            # Mengatur teks persentase dan label di luar chart
            fig2.update_traces(textinfo='percent+label', textfont_size=14, pull=[0.1] * len(labels))

            # Menambahkan total data di tengah donut chart
            fig2.add_annotation(
                dict(
                    font=dict(size=20, color="black"),
                    x=0.5,  # Posisi horizontal
                    y=0.5,  # Posisi vertikal
                    showarrow=False,
                    text=f'Total: {total_data}',  # Menampilkan total data
                    align="center"
                )
            )

            # Mengatur layout dan tampilan grafik
            fig2.update_layout(font=dict(size=14),
                            title_font=dict(size=20, family='Arial'),
                            showlegend=False)

            # Menampilkan grafik
            st.plotly_chart(fig2)
            # Menampilkan data hasil prediksi vs label asli
            df_test = pd.DataFrame({
                'Text': df['tweet_preprocessed'].values[:len(y_test)],  
                'Label Asli': y_test,
                'Prediksi Sentimen': y_pred
            })
            st.title("Klasifikasi Sentimen (Label Asli vs Prediksi Sentimen):")
            st.dataframe(df_test)

        else:
            st.error("CSV harus memiliki kolom 'tweet_preprocessed' dan 'label'.")
