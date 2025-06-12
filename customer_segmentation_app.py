# customer_segmentation_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt

# Page configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🧠 Customer Segmentation Using K-Means Clustering")
st.markdown("""
This app segments customers from the **Online Retail dataset** using **RFM (Recency, Frequency, Monetary) analysis** and **K-Means Clustering**.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Online Retail CSV File", type=["csv"], key="file_csv")


if uploaded_file is not None:
    # Load data
    df = pd.read_csv("C:\\Users\Tosha\\Customer Segmentation project for Online Retail\\Online Retail.csv")
    
    # Convert InvoiceDate to datetime with custom format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%y %H.%M', errors='coerce')

    # Optional: Drop rows where InvoiceDate couldn't be parsed
    df = df.dropna(subset=['InvoiceDate'])

    
    # Data Cleaning
    df = df[df['CustomerID'].notnull()]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df.drop_duplicates()

    # RFM Feature Engineering
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    latest_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                 # Frequency
        'TotalPrice': 'sum'                                     # Monetary
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    scaled_df = pd.DataFrame(scaled, columns=['Recency', 'Frequency', 'Monetary'])

    # K-Means Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled_df)

    # Assign human-readable segment names
    cluster_labels = {
        0: 'Champions',
        1: 'At Risk',
        2: 'Lost Customers',
        3: 'Potential Loyalist',
        4: 'New Customers'
    }
    rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

    # Show RFM Table
    st.subheader("🧮 RFM Table")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary']].head(10))

    # Cluster Summary
    st.subheader("📈 Cluster Summary")
    summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].agg(['mean', 'count']).round(1)
    st.dataframe(summary)

    # Visualizations
    st.subheader("📌 Visualizations")

    # Recency vs Frequency
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Segment', palette='Set2', ax=ax1)
    ax1.set_title('Recency vs Frequency')
    st.pyplot(fig1)

    # Frequency vs Monetary
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Segment', palette='Set2', ax=ax2)
    ax2.set_title('Frequency vs Monetary')
    st.pyplot(fig2)

    # Segment Exploration
    st.subheader("🔍 Explore Segment Details")
    selected_segment = st.selectbox("Select a Segment", rfm['Segment'].unique())
    st.dataframe(
        rfm[rfm['Segment'] == selected_segment].sort_values(by='Monetary', ascending=False).head(10)
    )

    # Download Button
    st.subheader("⬇️ Download Segmented Customer Data")
    csv = rfm.to_csv(index=False)
    st.download_button("Download as CSV", data=csv, file_name="customer_segments.csv", mime='text/csv')

else:
    st.info("👈 Upload the Online Retail Excel file to begin.")
