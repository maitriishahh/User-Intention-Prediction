# 🛒 User Purchase Intention Prediction

An end-to-end **Machine Learning Web App** that predicts whether a user is likely to make a purchase based on browsing behavior.
The project is fully deployed using **Streamlit (HuggingFace Spaces)** and **FastAPI (Render)**.

## 🚀 Live Demo

### 🌐 Streamlit App (Frontend)

👉 https://huggingface.co/spaces/maitrishah01/user-intention-ui

### ⚡ API Endpoint (Backend)

👉 https://user-intention-api.onrender.com


## 📌 Project Overview

This application predicts **User Purchase Intention** based on:

* Number of pages visited
* Product browsing behavior
* Bounce rate
* Exit rate
* Visitor type
* Month of visit
* Purchase intent score

The model provides:

* ✅ Purchase / Non-Purchase Prediction
* ✅ Probability Score
* ✅ Interactive UI

## 🧠 Architecture
```
User → Streamlit UI (HuggingFace)
      ↓
FastAPI Backend (Render)
      ↓
Machine Learning Model
```

## 🛠️ Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit

### Deployment

* HuggingFace Spaces (Frontend)
* Render (Backend)


## 📊 Features

* Interactive sliders for user behavior
* Real-time prediction
* Probability output
* Clean UI design
* API-based architecture
* Production deployment


## 📁 Project Structure

```
User-Intention-Prediction
│
├── artifacts/
├── user_intention_prediction/
├── main.py
├── app.py
├── requirements.txt
└── README.md
```


## ⚙️ How to Run Locally

### Clone Repository

```
git clone https://github.com/maitriishahh/User-Intention-Prediction.git
cd User-Intention-Prediction
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run FastAPI

```
uvicorn main:app --reload
```

### Run Streamlit

```
streamlit run app.py
```


## 🎯 Model Performance

* Algorithm: Random Forest / ML Model
* Task: Binary Classification
* Output: Purchase / No Purchase


## 💡 Future Improvements

* Add feature importance visualization
* Add probability gauge chart
* Add batch prediction support
* Add user session tracking


## 📸 Screenshots

<img width="1825" height="813" alt="Screenshot 2026-04-10 192303" src="https://github.com/user-attachments/assets/0d827c81-4f0c-4255-80e3-bce1528ce4bd" />
<img width="1850" height="806" alt="Screenshot 2026-04-10 192332" src="https://github.com/user-attachments/assets/20f7135c-0e2b-45b7-8065-6b5e48c18605" />


## 👩‍💻 Author

**Maitri Shah**


## ⭐ If you like this project

Give it a ⭐ on GitHub!
