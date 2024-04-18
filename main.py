import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px



st.write("""
FINA ANRIANI (2209116051)
""")

# Sidebar untuk navigasi
with st.sidebar:
    selected = option_menu('Menu',
                           ['Home',
                            'Data Visualization',
                            'Clustering'],

                            icons = ['house-heart', 
                                     'image-fill',
                                     'hearts'],
                            default_index = 0)



# Home page
if selected == 'Home':

    #page tittle
    # Menampilkan gambar dari file lokal
    from PIL import Image
    image = Image.open('dataset-cover.jpg')

    # Menampilkan gambar dengan ukuran yang disesuaikan
    st.image(image, caption='', use_column_width=True)

    st.subheader('Menganalisis Faktor-faktor yang Mempengaruhi seseorang Berbelanja di Online Store')
    st.caption('''Secara umum, dataset "Online Store Customer data" adalah kumpulan informasi atau data yang dikumpulkan dari pelanggan sebuah toko online. Dataset ini mencakup berbagai aspek terkait dengan interaksi pelanggan dengan toko online tersebut, seperti informasi pribadi, riwayat transaksi, perilaku belanja, feedback, dan lainnya.
    ''')
    st.caption('''Tujuan dari dataset ini adalah untuk memberikan wawasan atau pemahaman yang lebih baik tentang perilaku, kebutuhan, dan preferensi pelanggan, sehingga perusahaan atau bisnis dapat meningkatkan pengalaman pelanggan, retensi, dan efektivitas strategi pemasaran dan penjualan mereka.''')
    st.caption('Berikut ini merupakan link dari dataset yang digunakan: https://www.kaggle.com/datasets/mountboy/online-store-customer-data')
    
    
    st.write('DATASET AWAL')
    # Read data
    data = pd.read_csv('online_store_customer_data.csv')
    st.write (data)
    st.caption('Dataset awal ini adalah data mentah yang diperoleh langsung dari sumbernya tanpa melalui proses apapun. Dataset ini memiliki struktur, format, dan kualitas yang bervariasi tergantung dari sumber data aslinya.')
    st.caption('Dataset ini biasanya memerlukan pembersihan dan transformasi lebih lanjut sebelum dapat digunakan untuk analisis atau aplikasi lainnya.')

    st.write('DATASET AKHIR')
    df = pd.read_csv('online-store-customer-data-cleaned.csv')
    st.write (df)
    st.caption('Dataset akhir ini adalah data yang telah melalui proses pembersihan (cleaned) dan transformasi untuk memastikan kualitas, konsistensi, dan integritas data yang lebih baik.')
    st.caption('Dataset akhir ini memiliki struktur dan format yang konsisten, data yang lengkap dan tidak ada duplikat, serta siap untuk digunakan dalam analisis atau aplikasi lainnya. Dataset ini memudahkan analisis dan interpretasi data, serta menghasilkan insight dan hasil yang lebih akurat dan relevan.')

if selected == 'Data Visualization':
    df = pd.read_csv('online_store_customer_data.csv')
    # Visualisasi dengan Streamlit

    # 1. Pie Chart untuk Gender Distribution
    st.subheader("1. Gender Distribution")
    st.write('Tujuan: Visualisasi ini menunjukkan distribusi jenis kelamin dari data pelanggan.')
    st.write('Interpretasi: Pie chart menunjukkan distribusi gender pelanggan di toko online. Dari diagram, terlihat bahwa 54,6% pelanggan adalah laki-laki dan 45,4% adalah perempuan.')
    st.write('Insight: Distribusi gender pelanggan di toko online ini relatif seimbang, dengan sedikit lebih banyak laki-laki dibandingkan perempuan. Hal ini menunjukkan bahwa produk dan layanan toko online ini menarik bagi kedua jenis kelamin.')
    st.write('Actionable Insight : Pastikan produk dan layanan toko online menarik bagi both laki-laki dan perempuan. Gunakan strategi pemasaran yang menargetkan both laki-laki dan perempuan. Lakukan analisis data pelanggan secara berkala untuk memantau distribusi gender pelanggan.')
    gender_counts = df["Gender"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(gender_counts, labels=["Male", "Female"], autopct="%1.1f%%")
    ax1.set_title("Gender Distribution")
    st.pyplot(fig1)

    # 2. Histogram untuk Age Distribution
    st.subheader("2. Age Distribution")
    st.write('Tujuan: Histogram ini menampilkan distribusi usia pelanggan.')
    st.write('Interpretasi: Grafik menunjukkan distribusi usia pelanggan di toko online. Dari grafik, terlihat bahwa sebagian besar pelanggan (45%) berada di usia 20-34 tahun. Kelompok usia lainnya yang cukup besar adalah 16-19 tahun (23%), 35-44 tahun (16%), dan 45-54 tahun (10%).')
    st.write('Insight: Toko online ini populer di kalangan anak muda, dengan hampir 68% pelanggan berusia di bawah 35 tahun. Hal ini menunjukkan bahwa produk dan layanan toko online ini sesuai dengan kebutuhan dan preferensi anak muda.')
    st.write('Actionable Insight: Pastikan produk dan layanan toko online menarik bagi anak muda. Gunakan strategi pemasaran yang menargetkan anak muda. Lakukan analisis data pelanggan secara berkala untuk memantau distribusi usia pelanggan.')
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Age"], bins=20)
    ax2.set_title("Age Distribution")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Number of Customers")
    st.pyplot(fig2)

    # 3. Bar Chart untuk Customer Segmentation
    st.subheader("3. Customer Segmentation")
    st.write('Tujuan: Bar chart ini menampilkan jumlah pelanggan untuk setiap segmen yang ada dalam data.')
    st.write('Interpretasi: Grafik ini menunjukkan bahwa distribusi pelanggan berdasarkan kategori segmentasi di toko online tersebut didominasi oleh segmen Basic dengan persentase 42%. Kelompok segmen lainnya yang cukup signifikan adalah Silver (31%), Platinum (18%), dan Gold (9%).')
    st.write('Insight: Toko online ini lebih populer di kalangan pelanggan dengan segmen Basic dan Silver. Hal ini menunjukkan bahwa produk dan layanan yang ditawarkan oleh toko online tersebut sesuai dengan kebutuhan dan preferensi pelanggan di segmen tersebut.')
    st.write('Actionable Insight: Pastikan produk dan layanan menarik bagi pelanggan di segmen Basic dan Silver. Gunakan strategi pemasaran yang menargetkan pelanggan di segmen Basic dan Silver. Tawarkan promo menarik untuk menarik minat pelanggan di segmen Basic dan Silver')
    segment_counts = df["Segment"].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(segment_counts.index, segment_counts.values)
    ax3.set_title("Customer Segmentation")
    ax3.set_xlabel("Segment")
    ax3.set_ylabel("Number of Customers")
    st.pyplot(fig3)

    # 4. Pie Chart untuk Customer Online Store
    st.subheader("4. Customer Online Store")
    st.write('Tujuan: Visualisasi ini menunjukkan status pekerjaan dari pelanggan dalam konteks toko online.')
    st.write('Interpretasi: Grafik ini menunjukkan bahwa distribusi status status pekerjaan pelanggan yang berbelanja di toko online terkonsentrasi pada kategori "Self-employed" dengan persentase sebesar 38,1%. Hal ini menunjukkan bahwa sebagian besar pelanggan yang berbelanja di toko online merupakan pekerja mandiri. Kategorisasi status pekerjaan lainnya, seperti "Workers" (pekerja kantoran) dan "Unemployed" (pengangguran), memiliki persentase yang lebih kecil, yaitu 19,6% dan 10,3% masing-masing. Hal ini menunjukkan bahwa online store ini mungkin kurang populer di kalangan pekerja kantoran dan pengangguran.')
    st.write('Insight: Produk yang dijual di toko online menarik bagi pekerja mandiri, seperti produk-produk digital, produk-produk kreatif, atau produk-produk yang mendukung bisnis mereka. Pekerja mandiri lebih terbiasa berbelanja online karena mereka memiliki fleksibilitas waktu dan tempat untuk berbelanja.')
    st.write('Actionable Insight: Targetkan pemasaran ke pekerja mandiri. Tawarkan produk yang relevan.')
    employee_status_counts = df["Employees_status"].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.pie(employee_status_counts, labels=["Employees", "self-employee", "workers", "Unemployment"], autopct="%1.1f%%")
    ax4.set_title("Customer Online Store")
    st.pyplot(fig4)

    # 5. Histogram untuk Amount Spent Distribution
    st.subheader("5. Amount Spent Distribution")
    st.write('Tujuan: Histogram ini menampilkan distribusi jumlah yang dihabiskan oleh pelanggan.')
    st.write('Interpretasi: Grafik ini menunjukkan bahwa distribusi pengeluaran pelanggan online store terkonsentrasi pada kisaran Rp0 hingga Rp500.000. Hal ini menunjukkan bahwa sebagian besar pelanggan online store ini melakukan pembelian dengan nilai yang relatif kecil.')
    st.write('Insight: Sebagian besar pelanggan online store (lebih dari 50%) melakukan pembelian dengan nilai yang relatif kecil (Rp0 hingga Rp500.000).')
    st.write('Actionable Insight: Tawarkan harga yang kompetitif untuk produk-produk dengan nilai yang relatif kecil. Berikan promo dan diskon secara berkala untuk menarik minat pelanggan yang sensitif terhadap harga. Sediakan paket bundling produk yang dapat membantu pelanggan menghemat biaya.')
    fig5, ax5 = plt.subplots()
    ax5.hist(df["Amount_spent"], bins=20)
    ax5.set_title("Amount Spent Distribution")
    ax5.set_xlabel("Amount Spent")
    ax5.set_ylabel("Number of Customers")
    st.pyplot(fig5)

if selected == 'Clustering':
    st.subheader("Performing Data Clustering")
    
    num_clusters = st.slider("Number of Clusters", 2, 10, 3)

    data = pd.read_csv('online-store-customer-data-cleaned.csv')

    X = data[['Age', 'AmountSpentCategory']]
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(X)

    data['Cluster'] = clusters

    fig = px.scatter(data, x='Age', y='AmountSpentCategory', color='Cluster', 
                     title='Data Clustering: Consumer Spending by Age')
    st.plotly_chart(fig)

    st.write("Visualisasi ini menunjukkan Hubungan antara usia dan kategori pengeluaran mengacu pada analisis atau visualisasi yang menunjukkan bagaimana pola pengeluaran berbagai kelompok usia.")
    st.write("Ini bertujuan untuk memahami apakah ada korelasi atau pola tertentu antara rentang usia seseorang dan seberapa banyak mereka cenderung menghabiskan uang, baik dalam kategori pengeluaran rendah, sedang, atau tinggi.")
    st.write("Analisis ini dapat memberikan wawasan tentang perilaku belanja berdasarkan usia, yang dapat digunakan untuk strategi pemasaran, penargetan, dan pengembangan produk atau layanan.")