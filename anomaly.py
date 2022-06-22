#%%
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where
#%%
# # import data
# data = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
# df = data[["sepal_length", "sepal_width"]]
# %%

data = pd.read_csv('./MT_anomaly_detection.csv')
df = data[['out','in','coil']]
# model specification
model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.02).fit(df)
# %%
y_pred = model.predict(df)
y_pred
# %%
outlier_index = where(y_pred == -1) 
# filter outlier values
outlier_values = df.iloc[outlier_index]
outlier_values
# %%
plt.scatter(data["sepal_length"], df["sepal_width"])
plt.scatter(outlier_values["sepal_length"], outlier_values["sepal_width"], c = "r")
# %%



# %%
model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.03).fit(df)
# %%
y_pred = model.predict(df)
outlier_index = where(y_pred == -1) 


# %%
