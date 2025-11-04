# 🧠 Inventory Demand Forecasting Using Random Forest

An **AI-powered web application** that predicts inventory demand using **machine learning (Random Forest)**.  
The system helps businesses forecast product demand and make data-driven restocking decisions.

## 🚀 Features
- Flask-based backend with REST API  
- Random Forest regression model trained on real dataset  
- Displays model accuracy (R² score)  
- Interactive chart visualization using Chart.js  
- Input validation and responsive UI (Bootstrap)

## 🧩 Tech Stack
- **Backend:** Flask, Python, scikit-learn, pandas, joblib  
- **Frontend:** HTML, CSS (Bootstrap), JavaScript (Chart.js)  
- **ML Model:** Random Forest Regressor  

## 📊 Model Performance
- **R² Score:** ~99.97%  
- **MAE:** 0.11  
- **MSE:** 1.07  

## 🏗️ Project Structure

inventory-demand-forecasting-using-random-forest/
├── app.py # Flask web app
├── train.py # ML model training script
├── backend/ # Model + scaler + metadata files
├── static/ # CSS & JavaScript files
├── templates/ # HTML templates
└── data.csv # Dataset file  



## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/inventory-demand-forecasting-using-random-forest.git
   cd inventory-demand-forecasting-using-random-forest
2. Create and activate a virtual environment:
     python -m venv venv
     venv\Scripts\activate

 3.Install dependencies:
     pip install -r requirements.txt

4. Train the model:
      python train.py

5. Run the Flask app:
      python app.py

6. Open the app in your browser:
      http://127.0.0.1:5000/

7. 📸 Screenshot   

<img width="1750" height="835" alt="image" src="https://github.com/user-attachments/assets/4dbbcece-bc67-4079-bcf0-7cb9a9d47641" />

<img width="1812" height="856" alt="image" src="https://github.com/user-attachments/assets/b94111a3-62da-4770-af05-997696e1dc2f" />






