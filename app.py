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

col_header1, col_header2 = st.columns([1, 5]) # Rasio 1 untuk logo, 5 untuk teks

with col_header1:
    # Anda bisa menggunakan file lokal (misal: 'logo_uin.png') atau URL gambar
    st.image("images.png", width=150)

with col_header2:
    st.markdown("""
        <h1 style='margin-bottom: 0; color: #333333;'>Analisis Sentimen Menggunakan Algoritma Naïve Bayes</h1>
        <p style='font-size: 16px; color: #4B5563; font-style: italic;'>
        Studi Kasus: Pembelajaran Hybrid Learning di UIN Raden Intan Lampung
        </p>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #ebf0f5 !important;
        color: #333333 !important;
        padding: 20px !important;
    }
    .container {
        display: flex;
        justify-content: center; /*horizontal */
    }
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    .stTable, .stDataFrame, div[data-testid="stBlock"], 
    div[data-testid="stExpander .stAlert, [data-testid="element-container"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }
    /*PENGATURAN TEKS DAN UKURAN TABS */
    button[data-testid="stTabsTab"] {
        background-color: transparent !important;
    }
    button[data-testid="stTabsTab"] p {
        font-size: 30px !important;
        font-weight: 600 !important;
        color: #64748B !important;    /* Warna teks saat tidak aktif */
    }
    /* Ketika Tab sedang aktif diklik */
    button[aria-selected="true"] p {
        color: #1e8a25 !important;
        font-weight: bold !important;
        font-size: 30px !important;
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

    text = str(text).lower()
    text = replace_slang(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
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
st.markdown("""<p style="text-align: center;color: #2f3133; font-size: 18px;">
        Website analisis sentimen ini dikembangkan sebagai media implementasi dari metode Naïve Bayes dalam mengklasifikasikan data teks 
            ke dalam sentimen positif dan negatif. Website ini bertujuan untuk memudahkan proses analisis sentimen secara otomatis tanpa 
            memerlukan pengolahan data secara manual.
            </p>""", unsafe_allow_html=True)
st.markdown("""<p style="text-align: center;color: #2f3133; font-size: 18px;"> Pengguna dapat mengunggah data teks dalam format Excel, kemudian sistem akan melakukan 
            proses analisis dan menampilkan hasil klasifikasi sentimen, evaluasi serta visualisasi hasil analisis untuk memudahkan pemahaman pengguna
        </p>""", unsafe_allow_html=True)
st.markdown("---")

# upload file
uploaded_files = st.file_uploader(
        "Upload file excel (xlsx/xls)",
        type=['xlsx', 'xls'],
        accept_multiple_files=True
    )
if not uploaded_files:
    st.session_state['analisis_selesai'] = False
    st.session_state['daftar_hasil'] = []

if uploaded_files:
    st.info(f"Total berkas yang terdeteksi di sistem: {len(uploaded_files)} file.")
    if st.button("Jalankan Analisis Semua File"):
        with st.spinner("Sedang menjalankan analisis sentimen untuk seluruh file..."):
    
            kamus = pd.read_csv('kamus.csv', sep='\t')
            lexicon = dict(zip(kamus['word'], kamus['score']))
            
            # Reset container list lokal sebelum loop
            list_hasil_baru = []

            for i, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                df_awal = pd.read_excel(uploaded_file)
                 # cari semua kolom yang mengandung kata 'Opini'
                opini_cols = [col for col in df_awal.columns if 'opini' in col.lower()]

                if len(opini_cols) == 0:
                    st.error(f"File '{file_name}' tidak memiliki kolom 'Opini'!")
                    continue

                temp_name = f"_Opini_temp_{i}"
                while temp_name in df_awal.columns:
                    temp_name += "_1"

                # UBAH dari lebar ke panjang (baris)
                df_long = df_awal.melt(
                    id_vars=[col for col in df_awal.columns if col not in opini_cols],
                    value_vars=opini_cols,
                    var_name="Kolom Asal",
                    value_name=temp_name
                ).rename(columns={temp_name: "Opini"}).dropna(subset=["Opini"])

                df_long = df_long[df_long["Opini"].astype(str).str.strip() != ""].reset_index(drop=True)
                
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

                report_dict = classification_report(y_test, y_pred, output_dict=True)
                akurasi_file = report_dict['accuracy']

                # Masukkan seluruh objek hasil ke dictionary lokal khusus file ini
                data_file_ini = {
                    'nama': file_name,
                    'df_long': df_long.copy(), # .copy() memastikan data tidak terikat pointer loop
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'akurasi': akurasi_file
                    }
                    
                # Append ke list penampung
                list_hasil_baru.append(data_file_ini)
                
            # Pindahkan ke session state setelah seluruh file selesai di-looping total
            st.session_state['daftar_hasil'] = list_hasil_baru
            st.session_state['analisis_selesai'] = True
            st.success(f"Analisis selesai!")

st.markdown("---")

# --- LOGIKA PENAMPILAN TAB BERDASARKAN JUMLAH FILE ---
if st.session_state['analisis_selesai'] and st.session_state['daftar_hasil']:
    semua_hasil = st.session_state['daftar_hasil']
    num_files = len(semua_hasil)

    # KONDISI 1: JIKA HANYA ADA 1 FILE (Tampilan Sesuai Request Asli Anda)
    if num_files == 1:
        file_data = semua_hasil[0]
        nama_file = file_data['nama']
        df_long = file_data['df_long']
        y_test = file_data['y_test']
        y_pred = file_data['y_pred']

        tab1, tab2, tab3 = st.tabs(
            ["Preprocessing", "Visualisasi", "Evaluasi",]
            )
        with tab1:
                st.subheader("Contoh Hasil Preprocessing")
                st.dataframe(
                df_long[["Opini", "hasil_preprocessing", "label_sentimen"]]
                .rename(columns={
                    "hasil_preprocessing": "Hasil Preprocessing",
                    "label_sentimen": "Label Sentimen"
                    }).head())

        with tab2:
                # visualisasi pie chart
                st.subheader("Distribusi Sentimen")
                sentimen_counts = df_long['label_sentimen'].value_counts()
                total_data = sentimen_counts.sum()
                persentase = (sentimen_counts / total_data) * 100

                fig1, ax1 = plt.subplots(figsize=(4, 4))
                df_long['label_sentimen'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'salmon'], ax=ax1)
                ax1.set_ylabel("")
                st.pyplot(fig1, use_container_width=False)
                plt.clf()
                plt.close(fig_acc)

                #narasi hasil
                st.markdown(
                    f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Berdasarkan hasil visualisasi distribusi sentimen pada pie chart, 
                sentimen <b>{sentimen_counts.idxmax()}</b> mendominasi dengan persentase 
                sebesar <b>{persentase.max():.2f}%</b>, sedangkan sentimen lainnya memiliki 
                persentase sebesar <b>{persentase.min():.2f}%</b>. 
                Hal ini menunjukkan bahwa sebagian besar responden cenderung memberikan 
                respon <b>{sentimen_counts.idxmax().lower()}</b> terhadap topik yang dibahas.
                </div><br><br>""",unsafe_allow_html=True
                )

                # visualisasi bar chart
                st.subheader("Jumlah Data per Label Sentimen")
                sentimen_counts = df_long['label_sentimen'].value_counts()
                sentimen_counts.max()
                sentimen_counts.min()
                sentimen_counts.idxmax()   # positif
                sentimen_counts.idxmin()   # negatif

                fig2, ax2 = plt.subplots(figsize=(4, 3))
                sns.countplot(x='label_sentimen', data=df_long, palette='Set2', ax=ax2)
                ax2.set_ylabel("Jumlah")
                ax2.set_xlabel("Sentimen")       
                st.pyplot(fig2,use_container_width=False)
                plt.clf()
                plt.close(fig_acc)

                st.markdown(
                    f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Berdasarkan hasil visualisasi bar chart jumlah data per label sentimen, 
                diketahui bahwa sentimen <b>{sentimen_counts.idxmax()}</b> memiliki jumlah data 
                terbanyak yaitu sebanyak <b>{sentimen_counts.max()}</b> data, sedangkan sentimen 
                <b>{sentimen_counts.idxmin()}</b> memiliki jumlah data lebih sedikit yaitu 
                sebanyak <b>{sentimen_counts.min()}</b> data. 
                Perbedaan jumlah data ini menunjukkan adanya kecenderungan respon 
                yang lebih dominan terhadap salah satu sentimen.
                </div><br><br>""",unsafe_allow_html=True
                )

                # visualisasi wordcloud
                st.subheader("WordCloud Opini")
                all_text = ' '.join(df_long['hasil_preprocessing'])
                wc = WordCloud(width=800, height=400, max_words=80, background_color='white').generate(all_text)
                st.image(wc.to_array(), use_container_width=True)

                top_words = list(wc.words_.keys())[:5]
                top_words_str = ", ".join(top_words)

                st.markdown(
                    f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Berdasarkan hasil visualisasi word cloud dari seluruh opini, 
                kata-kata yang paling sering muncul antara lain 
                <b>{top_words_str}</b>. 
                Kata-kata tersebut mencerminkan topik yang dominan dibahas oleh responden. 
                Ukuran kata yang lebih besar pada visualisasi 
                menunjukkan frekuensi kemunculan yang lebih tinggi dalam data opini.
                </div><br><br>""",unsafe_allow_html=True
                )

                st.subheader("WordCloud Sentimen Positif & Negatif")
                positive_text = " ".join(df_long[df_long['label_sentimen'] == 'positif']['hasil_preprocessing'])
                wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)
                top_pos = ", ".join(list(wordcloud_pos.words_.keys())[:3])

                negative_text = " ".join(df_long[df_long['label_sentimen'] == 'negatif']['hasil_preprocessing'])
                wordcloud_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(negative_text)
                top_neg = ", ".join(list(wordcloud_neg.words_.keys())[:3])
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Sentimen Positif")
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud_pos, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                    plt.close(fig_acc)

                with col2:
                    st.markdown("### Sentimen Negatif")
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud_neg, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                    plt.close(fig_acc)

                st.markdown(
                    f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Berdasarkan word cloud sentimen positif, kata-kata yang paling sering muncul 
                antara lain <b>{top_pos}</b>, yang menunjukkan aspek baik atau bermanfaat 
                yang dirasakan oleh responden. 
                Sementara itu, pada word cloud sentimen negatif, kata-kata dominan yang muncul 
                antara lain <b>{top_neg}</b>, yang mencerminkan kendala atau permasalahan 
                yang dialami oleh responden. 
                Perbedaan kata dominan pada kedua word cloud ini menunjukkan adanya variasi 
                persepsi dalam opini yang ada.
                </div><br><br>""",unsafe_allow_html=True
                )

        with tab3:
                st.subheader("Evaluasi Model")
                st.text("Classification Report")
                st.text(classification_report(y_test, y_pred))

                report = classification_report(y_test, y_pred, output_dict=True)
                accuracy = report['accuracy']
                f1_pos = report['positif']['f1-score']
                f1_neg = report['negatif']['f1-score']

                st.markdown(
                    f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Hasil classification report menunjukkan bahwa model Naïve Bayes memperoleh 
                nilai akurasi sebesar <b>{accuracy:.2f}</b>. 
                Pada kelas sentimen positif, nilai F1-score yang diperoleh sebesar 
                <b>{f1_pos:.2f}</b>, sedangkan pada kelas sentimen negatif sebesar 
                <b>{f1_neg:.2f}</b>. 
                </div><br><br>""",unsafe_allow_html=True
                )

                # confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=["positif", "negatif"])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positif", "negatif"])
                fig, ax = plt.subplots(figsize=(4, 3))
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig, use_container_width=False)
                plt.clf()
                plt.close(fig_acc)

                TP = cm[0, 0]
                FN = cm[0, 1]
                FP = cm[1, 0]
                TN = cm[1, 1]
                total = TP + TN + FP + FN

                st.markdown(
                    f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Berdasarkan hasil confusion matrix, menunjukkan bahwa dari total <b>{total}</b> data uji, 
                <b>{TP}</b> data sentimen positif berhasil diklasifikasikan dengan benar 
                (True Positive), sementara <b>{FN}</b> data sentimen positif 
                salah diklasifikasikan sebagai sentimen negatif (False Negative). 
                Selanjutnya, sebanyak <b>{TN}</b> data sentimen negatif berhasil 
                diklasifikasikan dengan benar (True Negative), sedangkan <b>{FP}</b> data 
                sentimen negatif salah diklasifikasikan sebagai sentimen positif 
                (False Positive).
                </div>""",unsafe_allow_html=True
                )

                # download file
                excel_buffer = io.BytesIO()
                df_long[["Opini", "label_sentimen"]].to_excel(excel_buffer, index=False, sheet_name='Hasil Sentimen')
                excel_buffer.seek(0)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.download_button(
                    label="Download Hasil Labeling Excel",
                    data=excel_buffer,
                    file_name=f"hasil_{file_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# KONDISI 2: JIKA UPLOAD LEBIH DARI 1 FILE (Tampilan Berubah Jadi Per-File + perbandingan)
    else:
        nama_tab_files = [file_item['nama'] for file_item in semua_hasil]
        # Membuat nama tab
        tab_list = nama_tab_files + ["Perbandingan Komparatif"]
        tabs = st.tabs(tab_list)

        # Loop untuk mengisi masing-masing tab file
        for idx, file_item in enumerate(semua_hasil):
            with tabs[idx]:
                name = file_item['nama']
                df_file = file_item['df_long']
                yt = file_item['y_test']
                yp = file_item['y_pred']

                st.markdown(f"### Hasil Analisis Dokumen: **{name}**")

                with st.expander("Data Hasil Preprocessing", expanded=False):
                    st.dataframe(df_file[["Opini", "hasil_preprocessing", "label_sentimen"]].head(10))

                # Visualisasi
                col_chart1, col_chart2 = st.columns(2)
                counts = df_file['label_sentimen'].value_counts()
                
                with col_chart1:
                    st.subheader("Persentase Sentimen")
                    fig, ax = plt.subplots(figsize=(2, 2))
                    df_file['label_sentimen'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'salmon'], ax=ax)
                    ax.set_ylabel("")
                    st.pyplot(fig)
                    plt.clf()
                    plt.close(fig_acc)
                    
                with col_chart2:
                    st.subheader("WordCloud Dominan")
                    txt = ' '.join(df_file['hasil_preprocessing'])
                    if txt.strip():
                        wc_file = WordCloud(width=800, height=650, background_color='white').generate(txt)
                        st.image(wc_file.to_array())

                st.subheader("WordCloud Sentimen Positif & Negatif")
                pos_data = df_file[df_file['label_sentimen'] == 'positif']['hasil_preprocessing']
                positive_text = " ".join(pos_data) if not pos_data.empty else ""
                wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)
                top_pos = ", ".join(list(wordcloud_pos.words_.keys())[:3])

                neg_data = df_file[df_file['label_sentimen'] == 'negatif']['hasil_preprocessing']
                negative_text = " ".join(neg_data) if not neg_data.empty else ""
                wordcloud_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(negative_text)
                top_neg = ", ".join(list(wordcloud_neg.words_.keys())[:3])
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Sentimen Positif")
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud_pos, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                    plt.close(fig_acc)
                    st.text(f"Kata dominan positif pada file ini: {top_pos}")
                with col2:
                    st.markdown("### Sentimen Negatif")
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud_neg, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                    plt.close(fig_acc)
                    st.text(f"Kata dominan negatif pada file ini: {top_neg}")

                #Evaluasi Model
                with st.expander(" Evaluasi Model (Naïve Bayes)", expanded=False):
                    st.text("Classification Report:")
                    st.text(classification_report(yt, yp))
                    report_data = classification_report(yt, yp, output_dict=True)
                    accuracy = report_data['accuracy'] * 100
                    accuracy = report_data['accuracy'] * 100 # Diubah ke persen agar standar skripsi
                    f1_pos = report_data.get('positif', {}).get('f1-score', 0)
                    f1_neg = report_data.get('negatif', {}).get('f1-score', 0)

                    st.markdown(f"""
                    <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                    <b>Kesimpulan:</b><br>
                    Hasil Classification Report menunjukkan bahwa model memperoleh 
                    nilai akurasi keseluruhan sebesar <b>{accuracy:.2f}%</b>. 
                    Pada pengujian kelas sentimen positif, nilai F1-score yang diperoleh sebesar <b>{f1_pos:.2f}</b>, sedangkan pada kelas sentimen negatif sebesar <b>{f1_neg:.2f}</b>.
                    </div>
                    """, unsafe_allow_html=True)

                    # confusion matrix
                    st.text("Confusion Matrix:")
                    cm = confusion_matrix(yt, yp, labels=["positif", "negatif"])
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positif", "negatif"])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    disp.plot(ax=ax, cmap='Blues', values_format='d')
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    st.pyplot(fig, use_container_width=False)
                    plt.clf()
                    plt.close(fig_acc)

                    TP = cm[0, 0]
                    FN = cm[0, 1]
                    FP = cm[1, 0]
                    TN = cm[1, 1]
                    total = TP + TN + FP + FN

                    st.markdown(f"""
                    <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                    <b>Kesimpulan:</b><br>
                    Berdasarkan hasil confusion matrix untuk <b>{name}</b>, menunjukkan bahwa dari total <b>{total}</b> data uji, 
                    <b>{TP}</b> data sentimen positif berhasil diklasifikasikan dengan benar (True Positive), sementara <b>{FN}</b> data salah diklasifikasikan sebagai sentimen negatif (False Negative). 
                    Selanjutnya, sebanyak <b>{TN}</b> data sentimen negatif berhasil diklasifikasikan dengan benar (True Negative), sedangkan <b>{FP}</b> data salah diklasifikasikan sebagai sentimen positif (False Positive).
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tombol unduh hasil per file
                buf = io.BytesIO()
                df_file[["Opini", "label_sentimen"]].to_excel(buf, index=False)
                buf.seek(0)
                st.download_button(f"Download Hasil {name}", data=buf, file_name=f"hasil_{name}", key=f"dl_{idx}")
        
        # TAB TERAKHIR: PERBANDINGAN KOMPARATIF DARI SEMUA FILE
        with tabs[-1]:
            st.header("Perbandingan Komparatif Antar File")
            st.markdown("Berikut adalah komparasi sebaran sentimen positif vs negatif dari semua berkas dokumen yang Anda unggah.")
            
            # Membuat data rekapitulasi untuk dibandingkan
            compare_data = []
            for file_item in semua_hasil:
                name = file_item['nama']
                df_file = file_item['df_long']
                acc_val = file_item.get('akurasi', 0)
                counts = df_file['label_sentimen'].value_counts()
                pos_count = counts.get('positif', 0)
                neg_count = counts.get('negatif', 0)
                total = pos_count + neg_count
                
                compare_data.append({
                    "Nama File": name,
                    "Total Data": total,
                    "Sentimen Positif": pos_count,
                    "Sentimen Negatif": neg_count,
                    "% Positif": round((pos_count / total) * 100, 2) if total > 0 else 0,
                    "% Negatif": round((neg_count / total) * 100, 2) if total > 0 else 0,
                })
                
            df_compare = pd.DataFrame(compare_data)
            
            # Tabel Perbandingan
            st.subheader("Tabel Ringkasan Distribusi")
            st.dataframe(df_compare, use_container_width=True)
            
            col_grafik1, col_grafik2 = st.columns(2)

            with col_grafik1:
                st.subheader("Grafik Komparasi Persentase Sentimen")
                df_melted_perc = df_compare.melt(id_vars=["Nama File"], value_vars=["% Positif", "% Negatif"], 
                                            var_name="Sentimen", value_name="Persentase")
                
                fig_perc, ax_perc = plt.subplots(figsize=(10, 5))
                sns.barplot(x="Nama File", y="Persentase", hue="Sentimen", data=df_melted_perc, palette=['lightgreen', 'salmon'], ax=ax_perc)
                plt.xticks(rotation=15, ha='right')
                st.pyplot(fig_perc)
                plt.clf()
                plt.close(fig_acc)

                # Cari file paling positif dan paling negatif secara otomatis
                file_paling_pos = df_compare.loc[df_compare['% Positif'].idxmax()]['Nama File']
                persen_paling_pos = df_compare.loc[df_compare['% Positif'].idxmax()]['% Positif']
                file_paling_neg = df_compare.loc[df_compare['% Negatif'].idxmax()]['Nama File']

                idx_tertinggi_neg = df_compare["% Negatif"].idxmax()
                file_tertinggi_neg = df_compare.loc[idx_tertinggi_neg]["Nama File"]
                nilai_tertinggi_neg = df_compare.loc[idx_tertinggi_neg]["% Negatif"]

                rerata_persen_pos = df_compare["% Positif"].mean()
                rerata_persen_neg = df_compare["% Negatif"].mean()
                tren_dominan = "Positif" if rerata_persen_pos > rerata_persen_neg else "Negatif"
                nilai_dominan = max(rerata_persen_pos, rerata_persen_neg)
                st.markdown(f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Berdasarkan perbandingan data yang telah dilakukan diatas dapat disimpulkan bahwa 
                Dokumen dengan sentimen <b>Positif tertinggi</b> ditemukan pada file <b>{file_paling_pos}</b> dengan persentase mencapai <b>{persen_paling_pos:.2f}%</b>. 
                Sebaliknya, dokumen dengan sentimen <b>Negatif tertinggi</b> ditemukan pada file <b>{file_tertinggi_neg}</b> dengan persentase mencapai <b>{nilai_tertinggi_neg:.2f}%</b>.
                Secara Keseluruhan rata-rata persentase sentimen <b>{tren_dominan}</b> mendominasi dengan nilai sebesar <b>{nilai_dominan:.2f}%</b>.
                Anda bisa meninjau detail karakteristik masing-masing file pada tab-tab di atas.
                </div>
                """, unsafe_allow_html=True)

            with col_grafik2:
                st.subheader("Grafik Komparasi Akurasi Model")
                df_akurasi = pd.DataFrame({
                    "Nama File": [item['nama'] for item in semua_hasil],
                    "Akurasi": [item.get('akurasi', 0) for item in semua_hasil]
                    })
                fig_acc, ax_acc = plt.subplots(figsize=(5, 3))
                sns.barplot(x="Nama File", y="Akurasi", data=df_akurasi, palette='Blues', ax=ax_acc)
                for p in ax_acc.patches:
                    height = p.get_height()
                    ax_acc.annotate(
                    f"{height:.2f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', 
                    va='center', 
                    xytext=(0, 8),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold'
                )
                ax_acc.set_ylim(0, 1.1)
                ax_acc.set_ylabel("Nilai Akurasi")
                ax_acc.set_xlabel("Nama File")
                st.pyplot(fig_acc, use_container_width=False) 
                plt.clf()
                plt.close(fig_acc)

                idx_tertinggi_acc = df_compare["Akurasi Model"].idxmax() if "Akurasi Model" in df_compare.columns else df_akurasi["Akurasi"].idxmax()
                file_tertinggi_acc = df_compare.loc[idx_tertinggi_acc]["Nama File"] if "Akurasi Model" in df_compare.columns else df_akurasi.loc[idx_tertinggi_acc]["Nama File"]
                nilai_tertinggi_acc = (df_compare.loc[idx_tertinggi_acc]["Akurasi Model"] if "Akurasi Model" in df_compare.columns else df_akurasi.loc[idx_tertinggi_acc]["Akurasi"]) * 100
                st.markdown(f"""
                <div style="font-size:18px; line-height:1.6; background-color:#ddfcd9; padding: 15px; border-radius: 10px; color: #333333;">
                <b>Kesimpulan:</b><br>
                Evaluasi performa aglgoritma Naive Bayes dalam mengklasifikasikan data menunjukkan hasil yang bervariasi.
                Berdasarkan grafik komparasi, nilai akurasi tertinggi terdapat pada model klasifikasi <b>{file_tertinggi_acc}</b> dengan nilai akurasi sebesar <b>{nilai_tertinggi_acc:.2f}%</b>.
                Anda bisa meninjau detail karakteristik masing-masing file pada tab-tab di atas.
                </div>
                """, unsafe_allow_html=True)       
else:
    st.info("Silakan unggah satu atau beberapa file Excel di atas dan klik tombol 'Jalankan Analisis Semua File'.")

