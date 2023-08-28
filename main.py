import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score, f1_score, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
from io import StringIO

# Download data using requests
url = 'https://corgis-edu.github.io/corgis/datasets/csv/weather/weather.csv'
response = requests.get(url)
data = StringIO(response.text)
df = pd.read_csv(data)
df.head()

# Check if 'Data.Precipitation' column exists
if 'Data.Precipitation' not in df.columns:
    print("Data.Precipitation column does not exist in the DataFrame.")
    # Exit or handle error appropriately
    exit()
threshold = 1
df['RainTomorrow'] = df['Data.Precipitation'].apply(lambda x: 1 if x > threshold else 0)

# Data Preprocessing
columns_to_dummy = ['Data.Wind.Direction', 'Data.Precipitation']
columns_to_dummy = [col for col in columns_to_dummy if col in df.columns]
df_sydney_processed = pd.get_dummies(data=df, columns=columns_to_dummy)

#Training Data and Test Data
df_sydney_processed = df_sydney_processed.drop(columns=['Date.Full', 'Station.City', 'Station.Code', 'Station.Location', 'Station.State'])
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

# Split the dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=10)
print('Train set:', X_train.shape, Y_train.shape)
print('Test set:', X_test.shape, Y_test.shape)

#---------------------
# Linear Regression
# Creating and training the model
LinearReg = LinearRegression()
LinearReg.fit(X_train, Y_train)

# Getting predictions from the model
predictions = LinearReg.predict(X_test)

# Calculating the metrics
LinearRegression_MAE = mean_absolute_error(Y_test, predictions)
LinearRegression_MSE = mean_squared_error(Y_test, predictions)
LinearRegression_R2 = r2_score(Y_test, predictions)
print("Mean Absolute Error (MAE):", LinearRegression_MAE)
print("Mean Squared Error (MSE):", LinearRegression_MSE)
print("R^2 Score:", LinearRegression_R2)

# Creating a dataframe to display the metrics
report_data = {
    'Model': ['Linear Regression'],
    'MAE': [LinearRegression_MAE],
    'MSE': [LinearRegression_MSE],
    'R2': [LinearRegression_R2]
}
Report = pd.DataFrame(report_data)
# Display the dataframe
print(Report)

#---------------------
#KNN
# Create and train a KNN model
X_train = np.ascontiguousarray(X_train)
X_test = np.ascontiguousarray(X_test)
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(X_train, Y_train)

# Predict using the KNN model
predictions = KNN.predict(X_test)

# Calculate the performance metrics
KNN_Accuracy_Score = accuracy_score(Y_test, predictions)
KNN_JaccardIndex = jaccard_score(Y_test, predictions)
KNN_F1_Score = f1_score(Y_test, predictions)

# Print the metrics
print("KNN Accuracy Score:", KNN_Accuracy_Score)
print("KNN Jaccard Index:", KNN_JaccardIndex)
print("KNN F1 Score:", KNN_F1_Score)

#---------------------
#Decision Tree
#Create and train a Decision Tree model
Tree = DecisionTreeClassifier()
Tree.fit(X_train, Y_train)

#Predict using Decision Tree
predictions = Tree.predict(X_test)

#Calculate the accuracy, Jaccard index, and F1 score
Tree_Accuracy_Score = accuracy_score(Y_test, predictions)
Tree_JaccardIndex = jaccard_score(Y_test, predictions)
Tree_F1_Score = f1_score(Y_test, predictions)

# Printing the results
print("Decision Tree Accuracy Score:", Tree_Accuracy_Score)
print("Decision Tree Jaccard Index:", Tree_JaccardIndex)
print("Decision Tree F1 Score:", Tree_F1_Score)

#Logistic Regression
#Create and train a Logistic Regression model
LR = LogisticRegression(solver='liblinear')
LR.fit(X_train, Y_train)

#Predict using Decision Tree
predictions = LR.predict(X_test)

#Calculate the accuracy, Jaccard index, and F1 score
LR_Accuracy_Score = accuracy_score(Y_test, predictions)
LR_JaccardIndex = jaccard_score(Y_test, predictions)
LR_F1_Score = f1_score(Y_test, predictions)

# Log Loss
# For log loss, we need probability estimates of the positive class
predictions_prob = LR.predict_proba(X_train)[:, 1]
LR_Log_Loss = log_loss(Y_train, predictions_prob)

print("Logistic Regression Accuracy Score:", LR_Accuracy_Score)
print("Logistic Regression Jaccard Index:", LR_JaccardIndex)
print("Logistic Regression F1 Score:", LR_F1_Score)
print("Logistic Regression Log Loss:", LR_Log_Loss)

#---------------------
#SVM
#Create and train a Decision Tree model
SVM = svm.SVC()
SVM.fit(X_train, Y_train)

#Predict using SVM
predictions = SVM.predict(X_test)

#Calculate the accuracy, Jaccard index, and F1 score
SVM_Accuracy_Score = accuracy_score(Y_test, predictions)
SVM_JaccardIndex = jaccard_score(Y_test, predictions)
SVM_F1_Score = f1_score(Y_test, predictions)

print("SVM Accuracy Score:", SVM_Accuracy_Score)
print("SVM Jaccard Index:", SVM_JaccardIndex)
print("SVM F1 Score:", SVM_F1_Score)

#Report
report_data = {
    'Model': ['Linear Regression', 'KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
    'Accuracy Score': [LinearRegression_MAE, KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [np.nan, KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex], 
    'F1 Score': [np.nan, KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score],
    'Log Loss': [np.nan, np.nan, np.nan, LR_Log_Loss, np.nan]
}

Report = pd.DataFrame(report_data)
print(Report)
