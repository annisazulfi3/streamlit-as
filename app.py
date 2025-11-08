import streamlit as st
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
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
    #uah ke kata dasar
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
uploaded_file = st.file_uploader("Upload file excel (xlsx/xls) dengan kolom 'Opini'", type=['xlsx', 'xls'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    if 'Opini' not in df.columns:
        st.error("Kolom 'Opini' tidak ditemukan!")
    else:
        st.success("File berhasil diupload!")
        st.write("Contoh data:", df.head())

        # kamus Inset Lexicon
        kamus = pd.read_csv('kamus.csv', sep='\t')
        lexicon = dict(zip(kamus['word'], kamus['score']))

        # apply prapemrosesan dan pelabelan
        df['hasil_preprocessing'] = df['Opini'].apply(preprocess)
        
        df['label_sentimen'] = df['hasil_preprocessing'].apply(lambda x: label_sentiment(x, lexicon))

        st.subheader("Data setelah prapemrosesan & pelabelan")
        st.dataframe(df.head()[['hasil_preprocessing', 'label_sentimen']])

        # pembobotan & splitting
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['hasil_preprocessing'])
        y = df['label_sentimen']
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        except:
            st.warning("Data terlalu sedikit atau kelas tidak seimbang untuk stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # training testing
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # evaluasi
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
        st.pyplot(fig)

        # visualisasi pie chart
        st.subheader("Distribusi Sentimen")
        fig1, ax1 = plt.subplots()
        df['label_sentimen'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'salmon'], ax=ax1)
        st.pyplot(fig1)

        # visualisasi bar chart
        st.subheader("Jumlah Data per Label Sentimen")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='label_sentimen', data=df, palette='Set2', ax=ax2)
        st.pyplot(fig2)

        # visualisasi wordcloud
        st.subheader("WordCloud Opini")
        all_text = ' '.join(df['hasil_preprocessing'])
        wc = WordCloud(width=800, height=400, max_words=80, background_color='white').generate(all_text)
        st.image(wc.to_array(), use_container_width=True) 

        st.subheader("WordCloud Sentimen Positif & Negatif")
        positive_text = " ".join(df[df['label_sentimen'] == 'positif']['hasil_preprocessing'])
        wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)

        negative_text = " ".join(df[df['label_sentimen'] == 'negatif']['hasil_preprocessing'])
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
        df[['Opini', 'label_sentimen']].to_excel(excel_buffer, index=False, sheet_name='Hasil Sentimen')
        excel_buffer.seek(0)

        st.download_button(
            label="Download Hasil Labeling Excel",
            data=excel_buffer,
            file_name="hasil_sentimen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        )

