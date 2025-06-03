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
   - Load the dataset and view its basic information (e.g., number of rows, columns, data types).  
   - Examine summary statistics to understand distributions of numerical features.

2. **Visualizing Categorical Feature Distributions**  
   - Create bar plots or countplots to compare categories (e.g., education vs. loan_status).  
   - Analyze distributions of `gender`, `married`, `self_employed`, `property_area` against approval status.

3. **Correlation Analysis (Numerical Features)**  
   - Compute the correlation matrix for numeric columns (`applicant_income`, `coapplicant_income`, `loan_amount`, `loan_amount_term`, `credit_history`).  
   - Visualize the correlation matrix (e.g., using a heatmap) to identify strong relationships.

---

### Data Preprocessing

1. **Dropping Identifiers**  
   - Remove non-predictive ID columns (e.g., `loan_id`).

2. **Handling Missing Values**  
   - Fill missing values in numeric features (e.g., with median).  
   - Fill missing values in categorical features (e.g., with mode).

3. **Encoding Categorical Variables**  
   - Convert categorical columns (e.g., `gender`, `married`, `dependents`, `education`, `self_employed`, `property_area`, `loan_status`) into numeric labels.

4. **Splitting into Features and Target**  
   - Separate the target variable (`loan_status`) from the predictor features.

5. **Train/Test Split (Stratified)**  
   - Divide the data into training and testing sets, preserving the class distribution of the target.

6. **Standard Scaling**  
   - Standardize numeric features to have zero mean and unit variance.

---

### Principal Component Analysis (PCA)

1. **Compute Covariance Matrix and Eigenvalues**  
   - Analyze feature covariance to understand variance distribution.

2. **Calculate Explained Variance Ratio and Determine Number of Components**  
   - Identify how many principal components are needed to explain a desired proportion of variance.

3. **Apply PCA Transformation**  
   - Transform the standardized training and test data into the new PCA feature space.

---

### Model Training and Comparison

- **Classifiers Trained (with 10-fold cross-validation on PCA-transformed data):**  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Naive Bayes  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)

- **Evaluation Metric:**  
  - ROC AUC score computed for each classifier to compare performance.

---

### Hyperparameter Tuning

1. **Grid Search for SVM Parameters**  
   - Define a grid of candidate values for `C`, `kernel`, and (if applicable) `gamma`.  
   - Use cross-validation to select the best combination based on a chosen scoring metric (e.g., precision).

2. **Experiment Tracking with MLflow**  
   - Log hyperparameters, performance metrics, and model artifacts to MLflow for comparison and reproducibility.

---

### Final Evaluation

1. **Train Final Model**  
   - Retrain the best-performing classifier (e.g., tuned SVM) on the full training set.

2. **Predict on Test Set**  
   - Generate predictions for the hold-out test data.

3. **Compute Performance Metrics**  
   - Evaluate the final model using metrics such as accuracy, precision, recall, and confusion matrix.

4. **Log Metrics and Artifacts to MLflow**  
   - Record final test metrics and any relevant plots (e.g., confusion matrix) as MLflow artifacts.

---

## Technologies

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Statsmodels  
- scikit-learn  
- XGBoost  
- MLflow  
- Jupyter Notebook  

---

## Usage

1. **Obtain the project**  
   - Sklonuj lub pobierz repozytorium na swój komputer.

2. **Uruchomienie Jupyter Notebook**  
   - Otwórz środowisko Jupyter Notebook i załaduj plik `Loan_approval_final.ipynb`.

3. **Praca w notatniku**  
   - W pierwszych sekcjach wykonaj eksploracyjną analizę danych (EDA) oraz inżynierię cech.  
   - Następnie kontynuuj uruchamianie komórek krok po kroku, aby wytrenować i ocenić modele.  
   - Jeśli korzystasz z MLflow, możesz porównywać przebiegi eksperymentów w interfejsie MLflow.

4. **Alternatywne uruchomienie skryptów**  
   - Przygotuj dane (oczyszczenie, transformacje, zapis do pliku CSV).  
   - Uruchom skrypt służący do strojenia hiperparametrów oraz treningu modelu (jeśli jest dostępny w folderze `scripts`).  
   - Monitoruj przebieg eksperymentów i wyniki w interfejsie MLflow.

---

## Contact

Projekt udostępniony jest na licencji otwartej.  
- LinkedIn: [michał-jaros-88572821a](https://www.linkedin.com/in/michał-jaros-88572821a/)  
- E-mail: michal.marek.jaros@gmail.com  
