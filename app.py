import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import io

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from indoNLP.preprocessing import replace_slang

#konfig awal
st.set_page_config(page_title="Analisis Sentimen", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1000px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

#NLP
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopwords.words('indonesian'))
custom_stopword = { 'yg', 'jg', 'sih', 'aja', 'kan', 'kayak', 'nih', 'nggak', 'gitu', 'ya'
                   ,'pas', 'iya', 'tau', 'kalo', 'pada', 'ga', 'si', 'karna', 'gmn'
                   'gimana', 'gimanagimana', 'sma', 'hehe', 'yaa', 'tuh', 
                     }
stop_words = stop_words.union(custom_stopword)

# prapemrosesan data
def preprocess(text):
    #lowercase
    text = str(text).lower()
    #normalisasi kata
    text = replace_slang(text)
    #hapus angka, tanda baca, simbol
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    #tokenisasi
    tokens = word_tokenize(text)
    #hapus stopword
    tokens = [word for word in tokens if word not in stop_words]
    #ubah ke kata dasar
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [word for word in tokens if word.strip() != ""]
    return ' '.join(tokens)

# Pelabelan data
def label_sentiment(text, lexicon):
    tokens = word_tokenize(text)
    score = sum([lexicon.get(word, 0) for word in tokens])
    if score > 0:
        return "positif"
    else:
        return "negatif"
    
# Headline
st.markdown("<h1 style='text-align: center;'>Analisis Sentimen Menggunakan Algoritma Naive Bayes</h1>", unsafe_allow_html=True)
st.markdown("---")

# upload file
uploaded_file = st.file_uploader(
        "Upload file excel (xlsx/xls) dengan kolom 'Opini'",
        type=['xlsx', 'xls']
    )
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # cari semua kolom yang mengandung kata 'Opini'
    opini_cols = [col for col in df.columns if 'opini' in col.lower()]

    if len(opini_cols) == 0:
        st.error("Tidak ada kolom yang mengandung kata 'Opini'!")
    else:
        st.success(f"Ditemukan {len(opini_cols)} kolom opini: {opini_cols}")

        temp_name = "_Opini_temp_12345"
        while temp_name in df.columns:
            temp_name += "_1"

        # id_vars = semua kolom selain kolom opini
        id_vars = [col for col in df.columns if col not in opini_cols]

        # UBAH dari lebar → panjang (baris)
        df_long = df.melt(
            id_vars=[col for col in df.columns if col not in opini_cols],
            value_vars=opini_cols,
            var_name="Kolom_Asal",
            value_name=temp_name
        )

        # ganti nama kolom sementara jadi 'Opini'
        df_long = df_long.rename(columns={temp_name: "Opini"})

        # hapus NA / kosong
        df_long = df_long.dropna(subset=["Opini"])
        df_long = df_long[df_long["Opini"].astype(str).str.strip() != ""]

        st.write("Contoh data:", df_long.head())

        if st.button("Jalankan Analisis"):
            with st.spinner("Sedang menjalankan analisis sentimen..."):

                # kamus Inset Lexicon
                kamus = pd.read_csv('kamus.csv', sep='\t')
                lexicon = dict(zip(kamus['word'], kamus['score']))

                # apply prapemrosesan dan pelabelan
                df_long['hasil_preprocessing'] = df_long['Opini'].apply(preprocess)
                df_long['label_sentimen'] = df_long['hasil_preprocessing'].apply(lambda x: label_sentiment(x, lexicon))
                
                # pembobotan & splitting
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(df_long['hasil_preprocessing'])
                y = df_long['label_sentimen']
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                except:
                    st.warning("Data terlalu sedikit atau kelas tidak seimbang untuk stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # training testing
                model = MultinomialNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # simpan ke session_state (PENTING!)
                st.session_state['df_long'] = df_long
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred

            st.success("Analisis selesai!")

tab1, tab2, tab3 = st.tabs(
    ["Preprocessing", "Evaluasi", "Visualisasi"]
)
with tab1:
    if 'df_long' in st.session_state:
        df_long = st.session_state['df_long']

        st.subheader("Contoh Hasil Preprocessing")
        st.dataframe(st.session_state['df_long'].head())
    else:
        st.info("Silakan upload data dan jalankan analisis.")

with tab2:
    if 'y_test' in st.session_state and "y_pred" in st.session_state:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']

        st.subheader("Evaluasi Model")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=["positif", "negatif"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positif", "negatif"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Belum ada hasil evaluasi.")

with tab3:
    if 'df_long' in st.session_state:
        df_long = st.session_state['df_long']

        # visualisasi pie chart
        st.subheader("Distribusi Sentimen")
        fig1, ax1 = plt.subplots()
        df_long['label_sentimen'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'salmon'], ax=ax1)
        st.pyplot(fig1, use_container_width=True)

        # visualisasi bar chart
        st.subheader("Jumlah Data per Label Sentimen")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='label_sentimen', data=df_long, palette='Set2', ax=ax2)
        st.pyplot(fig2,use_container_width=True)

        # visualisasi wordcloud
        st.subheader("WordCloud Opini")
        all_text = ' '.join(df_long['hasil_preprocessing'])
        wc = WordCloud(width=800, height=400, max_words=80, background_color='white').generate(all_text)
        st.image(wc.to_array(), use_container_width=True)

        st.subheader("WordCloud Sentimen Positif & Negatif")
        positive_text = " ".join(df_long[df_long['label_sentimen'] == 'positif']['hasil_preprocessing'])
        wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)

        negative_text = " ".join(df_long[df_long['label_sentimen'] == 'negatif']['hasil_preprocessing'])
        wordcloud_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(negative_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Sentimen Positif")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pos, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("### Sentimen Negatif")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neg, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

        # download file
        excel_buffer = io.BytesIO()
        df_long[['Opini', 'label_sentimen']].to_excel(excel_buffer, index=False, sheet_name='Hasil Sentimen')
        excel_buffer.seek(0)

        st.download_button(
            label="Download Hasil Labeling Excel",
            data=excel_buffer,
            file_name="hasil_sentimen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Visualisasi belum tersedia.")

