#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import permutations, product, combinations 

def num_pieces(num,length):
    ot = list(range(1,length+1))[::-1]
    all_list = []
    for i in range(length-1):
        n = random.randint(1, num-ot[i])
        all_list.append(n)
        num -= n
    all_list.append(num) 
    return all_list

#%%

colname = ['ent','exit','motor']

threshold = [90,100,100]
state_num =  [2500,2500,2500,2500]
min_value = [-50,-50,-50]
max_value = [230,230,230]

possible_solution = list(set(list(combinations([0,0,0,1,1,1],3))))
possible_solution.sort(key=lambda tup: (tup[0],tup[1],tup[2]))  # sorts in place

total_state = [list(set(list(permutations(poss_state)))) for poss_state in possible_solution]
total_state_value = list(range(1,len(total_state)+1))

total_df = []
for num, state_list in enumerate(total_state):
    state_value = total_state_value[num]

    num_generate = state_num[num]
    temp_piece = num_pieces(num_generate, len(state_list))

    total_temp_df = []
    for i, temp_state in enumerate(state_list):  
        temp_num = temp_piece[i]
        temp_list = []

        for state in temp_state:
            if state == True: 
                temp = np.random.randint(low = threshold[i], high = max_value[i] , size = (temp_num))
            elif state ==False: 
                temp = np.random.randint(low = min_value[i], high = threshold[i] , size = (temp_num))
            else: 
                print("the input should be 0 or 1")    
            temp_list.append(temp)     
        temp_df = pd.DataFrame(temp_list).T
        temp_df.columns = colname  
        temp_df['state'] = state_value  
        total_temp_df.append(temp_df)   
        df = pd.concat(total_temp_df) 
    total_df.append(df)

final_df = pd.concat(total_df)


# %%
