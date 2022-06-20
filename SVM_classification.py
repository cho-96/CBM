#%%
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

#%%
MT = pd.read_csv('D:/Users/Cho/Projects/CBM/data/MT_label.csv')
scaler = StandardScaler()
features_standardized = scaler.fit_transform(MT.iloc[:,:-1])

X_train, X_test, y_train, y_test = train_test_split(features_standardized, MT.iloc[:,-1], test_size=0.2, random_state=42)

X_train = np.array(X_train).reshape(-1,3)
y_train = np.array(y_train).reshape(-1,1)-1
X_test = np.array(X_test).reshape(-1,3)
y_test = np.array(y_test).reshape(-1,1)-1

svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)
model = svc.fit(X_train, y_train)
# %%
label_predict = model.predict(X_test)
cf = confusion_matrix(y_test, label_predict)

# %%
print(recall_score(y_test, label_predict, average='macro'))
print(precision_score(y_test, label_predict, average='macro'))
print(accuracy_score(y_test, label_predict))
# %%
