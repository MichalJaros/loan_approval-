Loan Approval Prediction Pipeline

A complete machine learning pipeline for predicting loan approval status. This project performs data loading, exploratory data analysis, preprocessing (including scaling and PCA), model comparison (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, KNN, SVM), hyperparameter tuning, and final evaluation.  

---

## Table of Contents

- [Project Aim](#project-aim)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
  - [Data Preprocessing](#data-preprocessing)  
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)  
  - [Model Training and Comparison](#model-training-and-comparison)  
  - [Hyperparameter Tuning](#hyperparameter-tuning)  
  - [Final Evaluation](#final-evaluation)  
- [Technologies](#technologies)  
- [Usage](#usage)  
- [Repository Structure](#repository-structure)  
- [Contact](#contact)  

---

## Project Aim

Build and compare multiple classification models to predict whether a loan application will be approved. The pipeline covers:

1. Loading the “loan_approval_dataset.csv” file  
2. Performing exploratory data analysis to understand feature distributions and relationships  
3. Handling missing values and encoding categorical variables  
4. Standard scaling and dimensionality reduction via PCA  
5. Training and cross-validating several classifiers:  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Naive Bayes  
   - K-Nearest Neighbors  
   - Support Vector Machine  
6. Tuning the SVM hyperparameters using GridSearchCV  
7. Evaluating the final model on a held-out test set  

---

## Dataset

- **Filename**: `loan_approval_dataset.csv`  
- **Description**: Contains historical loan applications with features such as applicant income, credit history, loan amount, education, etc., and a target variable `loan_status` indicating approval (`Y`) or denial (`N`).  
- **Columns** (after dropping `loan_id`):  
  - `gender`  
  - `married`  
  - `dependents`  
  - `education`  
  - `self_employed`  
  - `applicant_income`  
  - `coapplicant_income`  
  - `loan_amount`  
  - `loan_amount_term`  
  - `credit_history`  
  - `property_area`  
  - `loan_status` (target: `Y` = approved, `N` = denied)  

---

## Methodology

### Exploratory Data Analysis (EDA)

1. **Inspecting Data Structure**  
   ```python
   dataset = pd.read_csv('loan_approval_dataset.csv')
   dataset.info()
   dataset.describe()
2. **Visualizing Categorical Feature Distributions**
•	Bar plots of education vs. loan_status
•	Countplots for gender, married, self_employed, property_area
3. **Correlation Analysis (Numerical Features)**
•	Compute correlation matrix for numeric columns (applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history)
•	Plot heatmap with seaborn
**Data Preprocessing**
1.	Dropping Identifiers
2.	Handling Missing Values (e.g., using median for numerics, mode for categoricals)
3.	Encoding Categorical Variables
4.	Splitting into Features and Target
5.	Train/Test Split (Stratified)
6.	Standard Scaling
**Principal Component Analysis (PCA)**
1.	Compute Covariance Matrix and Eigenvalues
2.	Calculate Explained Variance Ratio and Determine Number of Components
3.	Apply PCA Transformation
**Model Training and Comparison**
Train and 10-fold cross-validate the following classifiers on the PCA-transformed training data:
•	Logistic Regression
•	Decision Tree
•	Random Forest
•	Naive Bayes
•	K-Nearest Neighbors (KNN)
•	Support Vector Machine (SVM)
**Hyperparameter Tuning**
Perform grid search for the best SVM parameters using GridSearchCV
Log all hyperparameters and metrics with MLflow
**Final Evaluation**
1.	Train Final Model
2.	Predict on Test Set
3.	Compute Performance Metrics
4.	Log Metrics and Confusion Matrix Plot to MLflow
**Technologies**
•	Python
•	NumPy
•	Pandas
•	Matplotlib
•	Seaborn
•	Statsmodels
•	scikit-learn
•	XGBoost
•	MLflow
•	Jupyter Notebook
**Usage**
1.	Clone the repository
2.	Launch Jupyter Notebook
3.	Open and run Loan_approval_final.ipynb
o	Perform EDA and feature engineering in the first sections.
o	Execute the cells in order to train and evaluate models.
o	View and compare MLflow experiment runs (if MLflow server is running).
4.	To run scripts directly
o	Preprocess data and save cleaned DataFrame as CSV
o	Execute scripts/train_model.py (if provided)
o	Check MLflow UI for experiment tracking
**Contact**
This project is released under an open-source license.
•	LinkedIn: michał-jaros-88572821a
•	E-mail: michal.marek.jaros@gmail.com
