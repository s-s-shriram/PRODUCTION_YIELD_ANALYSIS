# PRODUCTION_YIELD_ANALYSIS
Data-driven machine learning system for predicting and optimizing production yield in manufacturing environments using Python, visualization, and IoT integration.
# 📊Production Yield Analysis

A machine learning–based production analytics system designed to optimize manufacturing yield through real-time data analysis and visualization. This project demonstrates how industrial sensor data can be leveraged to uncover patterns, identify inefficiencies, and predict output yield using supervised learning models.

---

## 📌 Project Overview

Manufacturing processes often face challenges like unpredictable output, process inefficiencies, and inconsistent quality. This project addresses those challenges by analyzing historical and simulated sensor data (temperature, humidity, machine speed, material type, etc.) to:

- **Predict production yield** using machine learning (Random Forest)
- **Visualize trends and relationships** through 10 different plots
- **Identify key performance drivers** through correlation and feature importance
- **Provide real-time insights** using embedded data without external files

The final system is modular, interpretable, and extensible for integration into smart factory environments.

---

## 🚀 Features

- 📊 **Data Analysis & Visualization**
  - Correlation matrix, histogram, KDE plot, line chart, boxplot, pie chart, pairplot, scatter plot, violin plot, and bar chart

- 🤖 **Machine Learning**
  - Random Forest Regressor to predict production yield
  - Encoded categorical features for cleaner model input
  - Evaluation using R² and Mean Absolute Error

- 📁 **Self-contained Dataset**
  - Sample dataset embedded using `StringIO` for easy demo without file dependency

- 💡 **Business Use Case**
  - Designed for industrial environments to support production managers and plant supervisors with actionable insights.

---

## 🧠 **Use Cases**

- Smart factory analytics
- Real-time yield monitoring
- Preventive diagnostics for machine performance
- Production planning and optimization

---

## 📦**REQUIREMENTS**
  - pandas==1.5.3
  - numpy==1.24.3
  - matplotlib==3.7.1
  - seaborn==0.12.2
  - scikit-learn==1.2.2
  - plotly==5.15.0

---
## 🧰 Tech Stack Used

- **🖥️ Programming Languages:**
  - 🐍 Python
  - 🗄️ SQL

- **⚙️ Frameworks & Libraries:**
  - 📊 Pandas
  - 🤖 Scikit-learn
  - 🧠 TensorFlow

- **🗃️ Databases:**
  - 💾 MySQL
  - 🗂️ PostgreSQL

- **🛠️ Tools:**
  - 📓 Jupyter Notebook
  - 🐳 Docker
  - 🧰 Git
