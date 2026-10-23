import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib



# Load the Dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Display first five rows
print(df.head())

# Check rows and columns
print(df.shape)

# Check data types and information
df.info()

# Check missing values
print(df.isnull().sum())

# Show column names
print(df.columns)

# Check missing values in TotalCharges
print(df["TotalCharges"].isnull().sum())

# Convert TotalCharges into numbers
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')

# Check missing values again
print(df["TotalCharges"].isnull().sum())

# Display rows with missing values
df1 = df[df["TotalCharges"].isnull()]
print(df1[["tenure","TotalCharges"]])

# Fill Missing TotalCharges values with 0
df["TotalCharges"] =  df["TotalCharges"].fillna(0)

# Check missing values again
print(df["TotalCharges"].isnull().sum())

# Remove customerID column
df = df.drop("customerID", axis=1)

# Display column names
print(df.columns)

# Check dataset size
print(df.shape)

# Check Churn values
print(df["Churn"].value_counts())

# Check Churn percentages
print(df["Churn"].value_counts(normalize=True)* 100)

# Count Yes and No customers
print(df["Churn"].value_counts())

# Draw bar chart
df["Churn"].value_counts().plot(kind="bar")

# Give title and labels
plt.title("Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

# Show chart
plt.show()

# Check numerical columns
print("Tenure Information:")
print(df["tenure"].describe())

print("Monthly Charges Information:")
print(df["MonthlyCharges"].describe())

print("Total Charges Information:")
print(df["TotalCharges"].describe())


# Display customer gender
print("Gender:")
print(df["gender"].value_counts())

# Display contract types
print("Contract:")
print(df["Contract"].value_counts())

# Display internet service types
print("Internet Service:")
print(df["InternetService"].value_counts())

# Display payment methods
print("Payment Method:")
print(df["PaymentMethod"].value_counts())

# Check duplicate rows
print("Duplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Check duplicate rows again
print(df.duplicated().sum())

# Check dataset size
print(df.shape)

# Check categorial columns
print("Categorial Columns:")

categorial_columns = df.select_dtypes(include="object").columns

print(categorial_columns)

# One-Hot Encoding

# Convert categorial data into numerical data

df = pd.get_dummies(df, drop_first=True, dtype=int)

# Check the updated dataset
print(df.head())

# Check dataset shape
print("Dataset Shape:", df.shape)

# Check column names
print(df.columns)

## Train-Test Split

# Separate input and output

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the shape

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Import Logistic Regression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("Model training completed successfully!")

# Predict on test data
y_pred = model.predict(X_test)

# Import evaluation metrics

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Import Decision Tree

# Create the model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
dt_pred = dt_model.predict(X_test)

# Evaluate the model

dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy * 100, "%")


## Import Random Forest

# Create the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Check Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy * 100, "%")

# Three models classification report

print("Logistic Regression:")
print(classification_report(y_test, y_pred))

print("Decision Tree:")
print(classification_report(y_test, dt_pred))

print("Random Forest:")
print(classification_report(y_test, dt_pred))

## Save the Model ##

joblib.dump(model, "customer_churn_model.pkl")

print("Model saved successfully!")

## Check input feature names
print(X.columns.tolist())










