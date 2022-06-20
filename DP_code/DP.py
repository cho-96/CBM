
# -----------------------------------------------------------------%
#  Condition based Optimization 
#
#  Module: CBM for Item-level Optimization
#  22:40 20th June 2022  
#    Reprogrammed by Jiung Kim 
#-----------------------------------------------------------------%
#%%
import numpy as np

def Item_opt(N, min_cond,r,perc): 
    """
    N: the number of products at each state 
    min_cond: minimum condition of each product for counting 
    r: ratio for probability metrix 
    perc: percentile for counting 
    """
    # P1 is one of the example probability matrix, this part should be done on hand
    P1 = np.array([0.69*r,0.31*r,0.00,0.00,0.00,
                   0.00,0.77*r,0.23*r,0.00,0.00,
                   0.00,0.00,0.92*r,0.08*r,0.00,
                   0.00,0.00,0.00,0.60*r,0.40*r,
                   0.00,0.00,0.00,0.00,1.00*r]).reshape(5,5) 
    assert len(N) == len(P1), ' the length should be equal ! '

    x = N
    T = 0 
    while True:
        x = np.dot(x,P1)
        ratio_x = x/np.sum(x)
        if ratio_x[-6+min_cond]>perc: 
            break
        T+=1
    return ratio_x, T
#%%
# example 
N = np.array([500,200,100,300,700])
min_cond = 5 # recommend 5  
r = 1 
perc = 0.5
_, T = Item_opt(N,min_cond,r,perc)

        
# %%
