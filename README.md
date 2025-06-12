# 🧠 Customer Segmentation Using K-Means Clustering

This Streamlit web app performs customer segmentation using **unsupervised learning** on the **Online Retail** dataset. It applies **RFM analysis (Recency, Frequency, Monetary)** and **K-Means clustering** to group customers into actionable segments.

---

## 📌 Features

✅ Upload a CSV version of the Online Retail dataset  
✅ Clean and preprocess transaction data  
✅ Create RFM features  
✅ Apply K-Means clustering with 5 segments  
✅ Label customer segments: Champions, At Risk, Lost Customers, etc.  
✅ Interactive visualizations (scatter plots, summaries)  
✅ Filter customers by segment  
✅ Download final segmented customer data as CSV  

---

## 📁 Folder Structure

Customer-Segmentation/
├── customer_segmentation_app.py # Main Streamlit App
├── Online Retail.csv # Sample dataset (keep size small)
├── README.md # Project documentation
└── .gitignore # Files to exclude from Git


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-segmentation-streamlit.git
cd customer-segmentation-streamlit


->Install Required packages
    pip install -r requirements.txt

->Run the app
    streamlit run customer_segmentation_app.py



📦 Dataset Format
Ensure your CSV includes:
.InvoiceNo
.StockCode
.Description
.Quantity
.InvoiceDate
.UnitPrice
.CustomerID
.Country


🛠 Tech Stack
-> Python
-> Pandas
-> Scikit-learn
-> Seaborn
-> Matplotlib
-> Streamlit

To install all the requirements, run "pip install -r requirements.txt" to the terminal

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Toshan Dahiya
Data Science & Analytics Enthusiast