# Data-Analize

**Analisis Sentimen Komentar Twitter Pemilu Menggunakan SVM**

## Library yang Perlu Diinstal:
1. Buka Command Prompt di Windows dan ketikkan perintah berikut:
    ```bash
    pip install streamlit pandas plotly wordcloud numpy pillow scikit-learn matplotlib seaborn nltk sastrawi streamlit-option-menu
    ```

2. **Catatan Penting:** Jika terjadi error saat menjalankan preprocessing data setelah stopword, tambahkan kode berikut pada baris 19:
    ```python
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    ```

## Akses Web:
1. Pindah ke direktori `web`:
    ```bash
    cd web
    ```

2. Jalankan aplikasi Streamlit:
    ```bash
    streamlit run app.py
    ```
