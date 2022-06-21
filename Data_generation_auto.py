#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import permutations, product, combinations 
from random import choices
import json

sensor = 'MT'

def num_pieces(num,length):
    final_list = [0]*length
    prob_list = [1/length]*length
    population = list(range(0,length))
    for _ in range(num):
        temp = np.random.choice(population,p = prob_list)
        final_list[temp]+=1
    return final_list

def level(colname,max_level): 
    num_col = len(colname)
    if num_col>=max_level:
        final_level = max_level
    else: 
        final_level = num_col+1
    return final_level

#%%

with open("D:/Users/Cho/Projects/CBM/data/sensors.json") as st_json:
    sensor_json = json.load(st_json)


colname = sensor_json[sensor]['colname'] 
coltype = sensor_json[sensor]['coltype'] 

threshold = sensor_json[sensor]['threshold'] 
final_level = level(colname,5)
total_num = 10000
state_num = num_pieces(total_num,final_level)
min_value = sensor_json[sensor]['min_value'] 
max_value = sensor_json[sensor]['max_value'] 

#%%
possible_solution = np.array([[0]*len(threshold)]*len(state_num))
for state in range(len(state_num)):
    for i in range(state):
        possible_solution[state][i]=1

total_state = [list(set(list(permutations(poss_state)))) for poss_state in possible_solution]
total_state_value = list(range(1,len(total_state)+1))
#%%
total_df = []
for num, state_list in enumerate(total_state):
    state_value = total_state_value[num]

    num_generate = state_num[num]
    temp_piece = num_pieces(num_generate, len(state_list))

    total_temp_df = []
    for i, temp_state in enumerate(state_list):  
        temp_num = temp_piece[i]
        temp_list = []

        for j, state in enumerate(temp_state):
            col_type = coltype[j]
            if col_type =='float':
                if state == True: 
                    temp = np.random.randint(low = threshold[j], high = max_value[j] , size = (temp_num))+np.random.rand()
                elif state ==False: 
                    temp = np.random.randint(low = min_value[j], high = threshold[j] , size = (temp_num))+np.random.rand()
                else: 
                    print("the input should be 0 or 1")
            else:
                if state == True: 
                    temp = [1]*temp_num
                elif state ==False: 
                    temp = [0]*temp_num
            temp_list.append(temp)     
        temp_df = pd.DataFrame(temp_list).T
        temp_df.columns = colname  
        temp_df['state'] = state_value  
        total_temp_df.append(temp_df)   
        df = pd.concat(total_temp_df) 
    total_df.append(df)

final_df = pd.concat(total_df)# %%

#%%
final_df.head()

#%%

final_df.to_csv('D:/Users/Cho/Projects/CBM/data/MT_label_{}.csv'.format(sensor), index=False)



# %%
