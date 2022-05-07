# First Task: Predict the percentage of a student based on the no. of study hours.

# Step 1 - Importing the dataset

# Importing all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore the warnings
import warnings as wg
wg.filterwarnings("ignore")

# Reading data from remote link

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)

# now let's observe the dataset
df.head()
df.tail()

# To find the number of columns and rows
df.shape

# To find more information about our dataset
df.info()

df.describe()

# now we will check if our dataset contains null or missings values
df.isnull().sum()
#############################################

# Step 2 - Visualizing the dataset

# Plotting the dataset
plt.rcParams["figure.figsize"] = [6,5]
df.plot(x='Hours', y='Scores', style='.', color='blue', markersize=10)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()

# We can also use .corr to determine the correlation between the variables
df.corr()
#############################################

# Step 3 - Data preparation

df.head()

# using iloc function we will devide the data
X = df.iloc[:, :1].values
y = df.iloc[:, 1:].values

# Splitting data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

#############################################

# Step 4 - Training the Algorithm

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
#############################################

# Step 5 - Visualizing the model

line = model.coef_*X + model.intercept_

# Plotting for the training data
plt.rcParams["figure.figsize"] = [6,5]
plt.scatter(X_train, y_train, color='blue')
plt.plot(X, line, color='black');
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()

# Plotting for the testing data
plt.rcParams["figure.figsize"] = [6,5]
plt.scatter(X_test, y_test, color='blue')
plt.plot(X, line, color='black');
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
#############################################

# Step 6 - Making predictions

print(X_test)   # Testing data - In Hours
y_pred = model.predict(X_test)   # Predicting the scores

# Comaring Actual vs Predicted

comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })

# Testing with your own data

hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for", hours, "hours is", own_pred[0])
#############################################

# Step 7 - Evaluating the model

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))






