#%%
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
#%%
MT = pd.read_csv('./MT_label3.csv')
col_len = len(MT.columns)-1
scaler = StandardScaler()
features_standardized = scaler.fit_transform(MT.iloc[:,:-1])

X_train, X_test, y_train, y_test = train_test_split(features_standardized, MT.iloc[:,-1], test_size=0.2, random_state=42)

X_train = np.array(X_train).reshape(-1,col_len)
y_train = np.array(y_train).reshape(-1,1)-1
X_test = np.array(X_test).reshape(-1,col_len)
y_test = np.array(y_test).reshape(-1,1)-1

#%%

# for tuning SVC
# parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100]}
# grid_svm = GridSearchCV(svc,
#                       param_grid = parameters, cv = 5)

# grid_svm.fit(X_train, y_train.squeeze())
# result = pd.DataFrame(grid_svm.cv_results_['params'])
# result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
# result.sort_values(by='mean_test_score', ascending=False)

svc = SVC(kernel="rbf", class_weight="balanced", C = 50, random_state=0)
model = svc.fit(X_train,y_train.squeeze())



# %%
label_predict = model.predict(X_test)
cf = confusion_matrix(y_test, label_predict)

# %%
print(recall_score(y_test, label_predict, average='macro'))
print(precision_score(y_test, label_predict, average='macro'))
print(accuracy_score(y_test, label_predict))
# %%
