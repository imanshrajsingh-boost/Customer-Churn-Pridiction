# 🔮 Customer Churn Prediction App

An end-to-end Machine Learning web application that predicts whether a customer is likely to churn (leave a business) based on their profile, usage frequency, subscription details, and billing patterns. 

Built with **Python**, **Scikit-Learn**, and **Streamlit**, this project demonstrates a complete data science workflow—from preprocessing and handling imbalanced datasets to hyperparameter tuning and model deployment.

Live Kaggle Notebook: [Churn Prediction Portfolio](https://www.kaggle.com/code/anshrajsingh7/churn-prediction)

---

## 🚀 Live Demo

👉 [Click here to view the Web App][(your-streamlit-link-goes-here)](https://customer-churn-pridiction-by-me.streamlit.app/#customer-churn-prediction-app)

---

## 📊 Business Problem & Dataset
Customer churn is a critical metric for any subscription-based business. Acquiring a new customer is often much more expensive than retaining an existing one. This app analyzes customer features to flags high-risk accounts so businesses can take proactive retention measures.

The model is trained on a customer dataset featuring details such as:
* **Demographics:** Age, Gender
* **Account Info:** Tenure, Subscription Type, Contract Length, Total Spend
* **Activity Metrics:** Usage Frequency, Support Calls, Payment Delay, Last Interaction

---

## 🛠️ The Machine Learning Workflow

1. **Data Cleaning & Exploration:** Handled missing values and analyzed feature distributions. Identified that customers with *Monthly* contracts and *higher support calls* had the highest churn rates.
2. **Feature Engineering:** * **Categorical Encoding:** Applied `OneHotEncoder` for Gender and `OrdinalEncoder` for structural categorical data (Subscription Type and Contract Length).
   * **Feature Scaling:** Used `StandardScaler` to normalize continuous variables to prevent large numbers from biasing the algorithms.
3. **Handling Imbalance:** Since the dataset was highly skewed (more churned examples), **`RandomUnderSampler` (RUS)** was applied to balance classes and optimize the model for better recall.
4. **Model Selection & Tuning:** Evaluated multiple baseline models (`Logistic Regression`, `Naive Bayes`, `Decision Trees`, `Random Forest`) and advanced gradient boosters (`XGBoost`, `LightGBM`, `CatBoost`).
5. **Final Selection:** A tuned **Random Forest Classifier** achieved the best overall **F1-Score (~0.95)** and Final Test Accuracy of **94%**.

---

## 💻 Tech Stack Used

* **Frontend:** Streamlit (For building the Interactive Web UI)
* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost, CatBoost, Imbalanced-Learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Model Storage:** Joblib

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit web application code
├── requirements.txt      # Python dependencies for deployment
├── Best_Model.pkl        # Serialized Tuned Random Forest model
├── Scaler.pkl            # Trained StandardScaler artifact
├── One_Hot_Encoder.pkl   # Saved OneHotEncoder configuration
├── Ordinal_Encoder.pkl   # Saved OrdinalEncoder configuration
└── README.md             # Project documentation
