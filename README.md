# Mobile Price Classification

## 1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models to predict the price range of mobile phones based on their technical specifications. The task is a multiclass classification problem where the mobile price is categorized into four distinct classes. The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment using a Streamlit web application.

## 2. Dataset Description
* **Dataset Name:** Mobile Price Classification
* **Source:** Kaggle
* **Original Dataset:** 2000 Instances
* **Data Split:**
    * **Training Data (`train.csv`):** 85% (Used for model training in .ipynb notebooks)
    * **Testing Data (`test.csv`):** 15% (Reserved for testing the Streamlit App)
* **Total Features:** 20
* **Target Variable:** `price_range`
* **Classes:**
    * 0 – Low cost
    * 1 – Medium cost
    * 2 – High cost
    * 3 – Very high cost

The dataset contains only numerical features and is perfectly balanced, making it suitable for evaluating different classification algorithms.
Within `train.csv`, an additional internal train–validation split was performed inside the Jupyter notebooks to evaluate model performance and compute the reported metrics.

## 3. Models Used and Evaluation Metrics
The following machine learning models were implemented using the same dataset and train-test split:
1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

### Performance Comparison
| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9588 | 0.9981 | 0.9592 | 0.9588 | 0.9590 | 0.9451 |
| **Decision Tree** | 0.7971 | 0.8647 | 0.7994 | 0.7971 | 0.7973 | 0.7300 |
| **kNN** | 0.5235 | 0.7716 | 0.5367 | 0.5235 | 0.5286 | 0.3655 |
| **Naive Bayes** | 0.8118 | 0.9470 | 0.8114 | 0.8118 | 0.8115 | 0.7491 |
| **Random Forest** | 0.8882 | 0.9767 | 0.8894 | 0.8882 | 0.8886 | 0.8511 |
| **XGBoost** | 0.8794 | 0.9866 | 0.8793 | 0.8794 | 0.8790 | 0.8394 |

## 4. Observations on Model Performance
| ML Model | Observation |
| :--- | :--- |
| **Logistic Regression** | **Best Performer.** Achieved the highest accuracy (~96%) and MCC, indicating the price range classes are linearly separable based on the provided features. |
| **Decision Tree** | Captured non-linear patterns but showed lower performance (79.7%) compared to ensemble methods, likely due to overfitting. |
| **kNN** | Performed poorly (52.3%) due to the high dimensionality of the data and sensitivity to feature scaling in a multiclass setting. |
| **Naive Bayes** | Delivered decent baseline results (81.1%) despite its assumption of feature independence. |
| **Random Forest** | Strong performance (~88.8%), significantly improving over the single Decision Tree by reducing variance through bagging. |
| **XGBoost** | Competitive performance (~87.9%) similar to Random Forest, demonstrating the effectiveness of gradient boosting, though Logistic Regression remained superior for this specific dataset. |

## 5. App Deployment

The interactive web application is deployed on Streamlit Community Cloud.

To comply with Streamlit Community Cloud resource constraints, all machine learning models are trained offline using `train.csv` and saved as pre-trained `.pkl` files.  
The Streamlit application loads these pre-trained models and performs prediction and evaluation **only** on the uploaded `test.csv`.

* **Live App Link:** https://mobile-price-classifier-raman.streamlit.app/

## 6. Project Structure

The project directory is organized as follows:

```text
ml-assignment-2/
├── app.py                 # Main Streamlit application
├── train.csv              # Training dataset (85%)
├── test.csv               # Testing dataset (15%)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
│
├── model/                 # Serialized models and scaler
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
└── notebook/              # Jupyter notebooks for training
    ├── logistic_regression.ipynb
    ├── decision_tree.ipynb
    ├── knn.ipynb
    ├── naive_bayes.ipynb
    ├── random_forest.ipynb
    └── xgboost.ipynb
