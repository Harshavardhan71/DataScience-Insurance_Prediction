# Insurance Purchase Prediction

Data Source: Kaggle

## Overview
This project focuses on predicting whether a customer will purchase an insurance policy based on their demographic and vehicle-related attributes. The dataset includes details such as age, gender, vehicle age, driving history, and policy details.

## Features
- **Data Preprocessing**: Loads and cleans training and test datasets.
- **Exploratory Data Analysis (EDA)**: Generates statistical insights and visualizations.
- **Feature Engineering**: Converts categorical variables to numerical values and performs feature selection.
- **Model Training**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Means Clustering
  - LightGBM
  - XGBoost with hyperparameter tuning using Optuna
  - Neural Network using TensorFlow/Keras
- **Model Evaluation**:
  - Accuracy & F1 Score computation
  - Confusion matrix visualization
  - Feature importance analysis

## AIM
This project is an **Insurance Purchase Prediction System** that aims to determine whether a customer will buy an insurance policy based on their demographic and vehicle-related characteristics. It follows a structured machine learning pipeline that includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and evaluation.

### **1. Data Loading & Preprocessing**
- Loads training and test datasets from a CSV file stored in Google Drive.
- Removes unnecessary columns (e.g., `id`).
- Converts categorical variables like `Gender`, `Vehicle_Age`, and `Vehicle_Damage` into numerical formats.
- Ensures data consistency and prepares it for analysis.

### **2. Exploratory Data Analysis (EDA)**
- Generates bar plots, histograms, and scatter plots using **Matplotlib, Seaborn, and Plotly** to understand data distributions.
- Analyzes customer demographics, driving history, and previous insurance status.
- Examines correlations between features and the target variable (`Response`).

### **3. Feature Engineering**
- Converts categorical data into numerical values.
- Identifies and selects the most relevant features based on correlation with the target variable.
- Drops redundant or less useful features like `Region_Code` and `Vintage` after analysis.

### **4. Model Training**
The project trains and evaluates multiple models to classify whether a customer will purchase insurance:

#### **4.1. Supervised Machine Learning Models**
- **Logistic Regression**: A baseline model for binary classification.
- **Decision Tree Classifier**: A tree-based model to capture complex relationships.
- **Random Forest Classifier**: An ensemble of decision trees to improve accuracy.
- **LightGBM Classifier**: A gradient boosting model optimized for speed and efficiency.
- **XGBoost Classifier**: Another gradient boosting model, fine-tuned using **Optuna** hyperparameter tuning.

#### **4.2. Unsupervised Learning for Clustering & Anomaly Detection**
- **K-Means Clustering**: Groups customers into two clusters to explore patterns.
- **COPOD Anomaly Detection**: Identifies anomalies in the dataset to improve model robustness.

#### **4.3. Deep Learning Model (Neural Network)**
- Implements a **Neural Network using TensorFlow/Keras** with:
  - Input layer with **7 features**.
  - Two hidden layers with **Batch Normalization** and **Dropout (0.3)**.
  - Output layer using **Softmax activation**.
  - Optimizer: **Adam** with a learning rate of **0.001**.
  - Uses weighted class balancing to handle imbalanced data.
  - Trains for **35 epochs** and evaluates performance.

### **5. Hyperparameter Tuning**
- Uses **Optuna** to find the best hyperparameters for **XGBoost** and **Random Forest**.
- Runs **500 trials** for XGBoost and **100 trials** for Random Forest to maximize **F1 Score**.

### **6. Model Evaluation**
- Evaluates all models using:
  - **Accuracy Score**: Measures overall correctness.
  - **F1 Score**: Balances precision and recall for imbalanced classes.
  - **Confusion Matrix**: Visualizes classification performance.
- Plots confusion matrices using **Seaborn**.

### **7. Insights & Results**
- The best models are selected based on their **F1 Score**.
- **Feature importance analysis** is performed to understand which factors influence predictions the most.
- Findings from EDA guide further improvements in feature engineering.

## **Project Achieves**
- Predicts insurance purchase likelihood based on customer data.  
- Compares multiple machine learning and deep learning models.  
- Uses **Optuna** for hyperparameter tuning to improve performance.  
- Visualizes data distributions, relationships, and model evaluations.  
- Provides insights into which features are most influential in customer decision-making.  

## Libraries Used
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Optuna, PyOD
- **Deep Learning**: TensorFlow, Keras

## Results
- The best-performing models were optimized using hyperparameter tuning.
- Insights from EDA helped in feature selection and engineering.

