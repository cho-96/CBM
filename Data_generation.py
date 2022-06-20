#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
arr1 = np.random.normal(loc=7, scale=10, size=(2000))
arr2 = np.random.normal(loc=16, scale=8, size=(2000))
arr3 = np.random.normal(loc=20, scale=5, size=(2000))
MT = {'ent': arr1, 'exit': arr2, 'motor': arr3}
label1 = pd.DataFrame(MT)
label1['state'] = 1
#%%
arr4 = np.random.normal(loc=15, scale=10, size=(2000))
arr5 = np.random.normal(loc=35, scale=12, size=(2000))
arr6 = np.random.normal(loc=40, scale=10, size=(2000))
MT2 = {'ent': arr4, 'exit': arr5, 'motor': arr6}
label2 = pd.DataFrame(MT2)
label2['state'] = 2
#%%
arr7 = np.random.normal(loc=20, scale=12, size=(2000))
arr8 = np.random.normal(loc=40, scale=10, size=(2000))
arr9 = np.random.normal(loc=60, scale=11, size=(2000))
MT3 = {'ent': arr7, 'exit': arr8, 'motor': arr9}
label3= pd.DataFrame(MT3)
label3['state'] = 3
#%%
arr10 = np.random.normal(loc=15, scale=13, size=(2000))
arr11 = np.random.normal(loc=63, scale=14, size=(2000))
arr12 = np.random.normal(loc=46, scale=12, size=(2000))
MT4 = {'ent': arr10, 'exit': arr11, 'motor': arr12}
label4 = pd.DataFrame(MT4)
label4['state'] = 4
#%%
arr13 = np.random.normal(loc=30, scale=15, size=(2000))
arr14 = np.random.normal(loc=70, scale=12, size=(2000))
arr15 = np.random.normal(loc=90, scale=13, size=(2000))
MT5 = {'ent': arr13, 'exit': arr14, 'motor': arr15}
label5 = pd.DataFrame(MT5)
label5['state'] = 5

#%%
MT_label = pd.concat([label1, label2, label3, label4, label5], ignore_index=True)
# %%
MT_label.tail()
# %%
MT_label[MT_label['ent']>=90]['ent'] = 90
MT_label[MT_label['exit']>=100]['exit'] = 100
MT_label[MT_label['motor']>=100]['motor'] = 100

#%%
MT_label.tail()
#%%
color_dict = {1:'red', 2:'blue', 3:'green', 4:'yellow', 5:'black'}

# %%
MT_label.replace({"state": color_dict})
# %%
plt.scatter(MT_label['ent'], MT_label['exit'], c=MT_label['state'])

# %%
MT_label.to_csv('D:/Users/Cho/Projects/CBM/data/MT_label.csv', index=False)
# %%
