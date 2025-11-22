

# Resume Screening App (Python + Streamlit)

This project is a machine-learning based Resume Screening Application built using Python and Streamlit.  
It analyzes resume datasets, trains multiple classification models, and displays results with visual charts.

---

## Features

- Resume dataset upload (CSV)
- Automated preprocessing and label encoding
- Multiple machine learning classification models
- Model comparison using accuracy and metrics
- Visual insights using Matplotlib and Seaborn
- Simple and interactive Streamlit UI

---

## Technologies and Libraries Used

### Data Handling
- pandas
- numpy

### Data Visualization
- matplotlib
- seaborn

### Machine Learning
- LabelEncoder
- K-Nearest Neighbors
- Support Vector Classifier
- Random Forest Classifier
- Logistic Regression
- Gaussian Naive Bayes
- One-Vs-Rest Classifier
- Accuracy Score
- Confusion Matrix
- Classification Report

### Framework
- Streamlit

---

## How the Application Works

1. User uploads a resume dataset (CSV).
2. Data is cleaned and label encoding is applied.
3. Data is split into training and testing sets.
4. Sparse matrices (if present) are converted to dense arrays.
5. Multiple ML models are trained:
   - KNN
   - SVC
   - Random Forest
   - Logistic Regression
   - Gaussian Naive Bayes
6. One-Vs-Rest strategy is used for multi-class classification.
7. Models are evaluated using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
8. Streamlit displays results and visual charts.

---

## Project Folder Structure

```

RESUME_SCREENING_APP
│
├── .ipynb_checkpoints
├── app.py
├── clf.pkl
├── encoder.pkl
├── readme.md
├── Resume_Screening_App.ipynb
├── tfidf.pkl
└── UpdatedResumeDataSet.csv

```

---

## How to Run the Project

### 1. Install Dependencies
```

pip install -r requirements.txt

```

### 2. Run the Streamlit Application
```

streamlit run app.py

```

### 3. Open in Browser
```

[http://localhost:8501/](http://localhost:8501/)

```

---

## Algorithms Used

### K-Nearest Neighbors (KNN)
Distance-based model that classifies resumes based on nearest points.

### Support Vector Classifier (SVC)
Finds an optimal hyperplane for separating resume categories.

### Random Forest Classifier
Uses an ensemble of decision trees for accurate predictions.

### Logistic Regression
Fast linear classifier for multi-class output.

### Gaussian Naive Bayes
Probabilistic model assuming independence between features.

### One-Vs-Rest (OVR)
Converts binary classifiers into multi-class classifiers.

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1)

---

## Visualizations

- Confusion matrix heatmaps
- Feature distribution plots
- Model comparison graphs
- Correlation heatmaps

---

## Future Enhancements

- Resume text extraction using NLP (spaCy/NLTK)
- BERT/Transformer-based classification
- Automated resume ranking system
- Exportable PDF reports
