import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc


data = pd.read_csv('/content/inpuu.csv')
data.head()
data.describe()

data.isnull().any()
data = data.replace("?", np.nan)
print(data)
# Impute missing values with the mean of the respective columns
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())
from sklearn.model_selection import train_test_split

X = data.iloc[:,:13].values
y = data["target"].values
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0 )
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Replace "?" with NaN to identify missing values

# Proceed with the rest of the code
# ...
data.hist(figsize = (12, 12))
plt.show()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = Sequential()
classifier.add(Dense(activation = "relu", input_dim = 13,
					units = 8, kernel_initializer = "uniform"))
classifier.add(Dense(activation = "relu", units = 14,
					kernel_initializer = "uniform"))
classifier.add(Dense(activation = "sigmoid", units = 1,
					kernel_initializer = "uniform"))
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',
				metrics = ['accuracy'] )
classifier.fit(X_train , y_train , batch_size = 8 ,epochs = 100 )
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test,y_pred)
cm
accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
print("the accuracy obtained is : ")
print(accuracy*100)
input_data = pd.read_csv('/content/true.csv')
input_data.head()
input_data.describe()

input_data.isnull().any()
input_data = input_data.replace("?", np.nan)
print(input_data)
# Impute missing values with the mean of the respective columns
input_data = input_data.fillna(input_data.mean())
input_data = input_data.apply(pd.to_numeric, errors='coerce')
input_data_features = input_data.iloc[:, :13].values

# Scale the input data using the same StandardScaler 'sc'
input_data_scaled = sc.transform(input_data_features)


# Make predictions
y_pred_input = classifier.predict(input_data_scaled)
y_pred_input = (y_pred_input > 0)
if not y_pred_input:
  print("this person is not having heart disease")
else:
  print("this person is having heart disease,kindly diagnose further")

y_pred_train = classifier.predict(X_train)
y_pred_train = (y_pred_train > 0.5)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot a pie chart for the distribution of predicted classes
labels = ['Negative', 'Positive']
sizes = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Predicted Class Distribution')
plt.show()