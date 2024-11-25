import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the data
diabetic_data = pd.read_csv("./data/diabetic_data.csv")
ids_mapping = pd.read_csv("./data/IDS_mapping.csv")

# Data Preprocessing
# Drop columns that won't help in prediction (e.g., unique patient IDs)
diabetic_data.drop(columns=["encounter_id", "patient_nbr"], inplace=True)

# Handle missing values
diabetic_data.replace("?", np.nan, inplace=True)
missing_values = diabetic_data.isnull().sum()
missing_columns = missing_values[missing_values > 0].index.tolist()
diabetic_data.drop(
    columns=missing_columns, inplace=True
)  # Drop columns with missing values

# Encode categorical variables
categorical_cols = diabetic_data.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    diabetic_data[col] = le.fit_transform(diabetic_data[col])

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.countplot(x="readmitted", data=diabetic_data)
plt.xticks(ticks=[0, 1, 2], labels=["No", "<30", ">30"])
plt.title("Distribution of Readmission Status")
plt.xlabel("Readmitted")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(diabetic_data["age"], bins=10, kde=True)
plt.xticks(
    ticks=range(len(le.classes_)), labels=le.inverse_transform(range(len(le.classes_)))
)
plt.title("Age Distribution of Patients")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(14, 10))
sns.boxplot(x="readmitted", y="time_in_hospital", data=diabetic_data)
plt.xticks(ticks=[0, 1, 2], labels=["No", "<30", ">30"])
plt.title("Time in Hospital vs Readmission Status")
plt.xlabel("Readmitted")
plt.ylabel("Time in Hospital (days)")
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(diabetic_data.corr(), annot=True, cmap="viridis", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Age and Gender based visualizations
plt.figure(figsize=(12, 8))
sns.countplot(x="age", hue="gender", data=diabetic_data)
plt.xticks(
    ticks=range(len(le.classes_)), labels=le.inverse_transform(range(len(le.classes_)))
)
plt.title("Age Group Distribution by Gender")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x="gender", hue="readmitted", data=diabetic_data)
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.title("Gender Distribution by Readmission Status")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(14, 10))
sns.boxplot(x="age", y="time_in_hospital", hue="gender", data=diabetic_data)
plt.xticks(
    ticks=range(len(le.classes_)), labels=le.inverse_transform(range(len(le.classes_)))
)
plt.title("Time in Hospital by Age and Gender")
plt.xlabel("Age Group")
plt.ylabel("Time in Hospital (days)")
plt.xticks(rotation=45)
plt.show()

# Split dataset into features and target variable
X = diabetic_data.drop(columns=["readmitted"])
y = diabetic_data["readmitted"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building
# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Support Vector Machine (SVM)
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# XGBoost Classifier
xgb_model = XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# AdaBoost Classifier
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

# Deep Learning Model
deep_model = Sequential()
deep_model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
deep_model.add(Dropout(0.5))
deep_model.add(Dense(32, activation="relu"))
deep_model.add(Dropout(0.5))
deep_model.add(Dense(1, activation="sigmoid"))

# Compile the model
deep_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
deep_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# Predict with Deep Learning Model
y_pred_deep = (deep_model.predict(X_test) > 0.5).astype("int32")


# Model Evaluation
def evaluate_model(model_name, y_test, y_pred):
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Evaluate Logistic Regression
evaluate_model("Logistic Regression", y_test, y_pred_logistic)

# Evaluate Random Forest
evaluate_model("Random Forest", y_test, y_pred_rf)

# Evaluate SVM
evaluate_model("Support Vector Machine (SVM)", y_test, y_pred_svm)

# Evaluate Gradient Boosting
evaluate_model("Gradient Boosting", y_test, y_pred_gb)

# Evaluate XGBoost
evaluate_model("XGBoost", y_test, y_pred_xgb)

# Evaluate AdaBoost
evaluate_model("AdaBoost", y_test, y_pred_ada)

# Evaluate Deep Learning Model
evaluate_model("Deep Learning Model", y_test, y_pred_deep)

# Comparing All Models
print(
    "Logistic Regression vs Random Forest vs SVM vs Gradient Boosting vs XGBoost vs AdaBoost vs Deep Learning Model:"
)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Deep Learning Model Accuracy:", accuracy_score(y_test, y_pred_deep))
