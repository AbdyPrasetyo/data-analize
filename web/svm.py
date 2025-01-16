# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# def show_svm():
#     # Fungsi untuk memuat model dan data
#     def load_data(file):
#         data = pd.read_csv(file, sep=';')  # Memuat CSV dengan pemisah koma
#         return data

#     # Fungsi untuk pelatihan dan prediksi
#     def classify_sentiment(texts, labels):
#         # Menggunakan TF-IDF untuk representasi teks
#         vectorizer = TfidfVectorizer()
#         X = vectorizer.fit_transform(texts)
        
#         # Memisahkan data untuk pelatihan dan pengujian
#         X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
#         # Membuat model SVM
#         model = SVC(kernel='linear')
#         model.fit(X_train, y_train)
        
#         # Memprediksi hasil
#         y_pred = model.predict(X_test)
        
#         # Menghitung hasil evaluasi
#         acc = accuracy_score(y_test, y_pred)  # Akurasi
#         conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion Matrix
#         report = classification_report(y_test, y_pred, output_dict=True)  # Classification report
        
#         return acc, conf_matrix, report, y_pred, y_test, X_test, X_train

#     st.title('Klasifikasi Sentimen')

#     uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

#     if uploaded_file is not None:
#         df = load_data(uploaded_file)
#         st.write("Data yang diunggah:", df)
        
#         if 'tweet_preprocessed' in df.columns and 'label' in df.columns:
#             # Melakukan klasifikasi
#             acc, conf_matrix, report, y_pred, y_test, X_test, X_train = classify_sentiment(df['tweet_preprocessed'], df['label'])
            
#             # Menampilkan hasil akurasi
#             st.write("Hasil Klasifikasi Sentimen:")
#             st.write(f"Akurasi Model: {acc * 100:.2f}%")
            
#             st.text(classification_report(y_test, y_pred))
            

#             # Menampilkan Confusion Matrix
#             st.write("Confusion Matrix:")
#             fig, ax = plt.subplots(figsize=(6, 4))
#             sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Positif', 'Negatif', 'Netral'], yticklabels=['Positif', 'Negatif', 'Netral'])
#             plt.ylabel('Aktual')
#             plt.xlabel('Prediksi')
#             st.pyplot(fig)
            

#             # Distribusi prediksi sentimen
#             st.write("Distribusi Prediksi Sentimen:")
#             pred_counts = pd.Series(y_pred).value_counts()
#             labels = pred_counts.index
#             sizes = pred_counts.values
         
#             total_data = sizes.sum()

        
#             fig2 = px.pie(names=labels, hole=0.7, 
#                         color_discrete_sequence=px.colors.qualitative.Pastel, 
#                         title="Distribusi Prediksi Sentimen")

#             # Mengatur teks persentase dan label di luar chart
#             fig2.update_traces(textinfo='percent+label', textfont_size=14, pull=[0.1, 0.1, 0.1])  # Menambahkan efek 'pull'

#             # Menambahkan total data di tengah donut chart
#             fig2.add_annotation(
#                 dict(
#                     font=dict(size=20, color="black"),
#                     x=0.5,  # Posisi horizontal
#                     y=0.5,  # Posisi vertikal
#                     showarrow=False,
#                     text=f'Total: {total_data}',  # Menampilkan total data
#                     align="center"
#                 )
#             )

#             # Mengatur layout dan tampilan grafik
#             fig2.update_layout(font=dict(size=14),
#                             title_font=dict(size=20, family='Arial'),
#                             showlegend=False)

#             # Menampilkan grafik
#             st.plotly_chart(fig2)

#             # Menampilkan tabel prediksi vs label asli
#             df_test = pd.DataFrame({
#                 'Text': df['tweet_preprocessed'].values[:len(y_test)],  
#                 'Label Asli': y_test,
#                 'Prediksi Sentimen': y_pred
#             })
#             st.write("Klasifikasi Sentimen (Label Asli vs Prediksi Sentimen):")
#             st.dataframe(df_test)

           

            
#         else:
#             st.error("CSV harus memiliki kolom 'tweet_preprocessed' dan 'label'.")


#     # Membuat donut chart untuk distribusi prediksi
#             # st.write("Distribusi Prediksi Sentimen:")
#             # pred_counts = pd.Series(y_pred).value_counts()
#             # labels = pred_counts.index
#             # sizes = pred_counts.values
#             # colors = sns.color_palette('pastel')[0:len(labels)]

#             # fig2, ax2 = plt.subplots()
#             # ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'linewidth': 0.7, 'edgecolor': 'white'})
      

#             # centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#             # fig2.gca().add_artist(centre_circle)
#             # total_data = sizes.sum()
#             # ax2.text(0, 0, f'{total_data}', ha='center', va='center', fontsize=14, fontweight='bold')

#             # plt.axis('equal')  # Memastikan chart berbentuk bulat
#             # st.pyplot(fig2)

import streamlit as st 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def show_svm():
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
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        # Membuat model SVM
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        
        # Memprediksi hasil
        y_pred = model.predict(X_test)
        
        # Menghitung hasil evaluasi
        acc = accuracy_score(y_test, y_pred)  # Akurasi
        conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion Matrix
        report = classification_report(y_test, y_pred, output_dict=True)  # Classification report
        
        return acc, conf_matrix, report, y_pred, y_test, X_test, X_train

    st.title('Klasifikasi Sentimen')

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data yang diunggah:", df)
        
        if 'tweet_preprocessed' in df.columns and 'label' in df.columns:
            # Melakukan klasifikasi
            acc, conf_matrix, report, y_pred, y_test, X_test, X_train = classify_sentiment(df['tweet_preprocessed'], df['label'])
              # Menampilkan distribusi panjang teks
            st.write("Distribusi Panjang Teks pada Dataset Pelatihan dan Pengujian:")
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
            # Menampilkan hasil akurasi
            st.write("Hasil Klasifikasi Sentimen:")
            st.write(f"Akurasi Model: {acc * 100:.2f}%")
            
            st.text(classification_report(y_test, y_pred))
            
          
            
            # Menampilkan Confusion Matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Positif', 'Negatif', 'Netral'], yticklabels=['Positif', 'Negatif', 'Netral'])
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            st.pyplot(fig)

            # Menampilkan donut chart untuk distribusi prediksi
            st.write("Prediksi SVM:")
            pred_counts = pd.Series(y_pred).value_counts()
            labels = pred_counts.index
            sizes = pred_counts.values
            total_data = sizes.sum()

            fig2 = px.pie(names=labels, hole=0.7, 
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

        else:
            st.error("CSV harus memiliki kolom 'tweet_preprocessed' dan 'label'.")
