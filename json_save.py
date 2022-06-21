#%%
import json

sensors_json = {"MT": {"colname": ['ent','exit','motor'],
                "coltype": ['float','float','float'],
                "threshold": [90,100,100],
                "state_num": [500,500,500,500],
                "min_value": [-50,-50,-50],
                "max_value": [229,229,229]
},"CM": {"colname": ['pressure','exit','motor'],
                "coltype": ['float','float','float'],
                "threshold": [5,200,40],
                "state_num": [500,500,500],
                "min_value": [0,-50,0],
                "max_value": [15,500,100]
},"HVAC": {"colname": ['out','in','high','low','coil','comp','cond'],
                "coltype": ['float','float','binary','binary','float','binary','binary'],
                "threshold": [50,50,0,0,40,0,0],
                "state_num": [2000,2000,2000,2000,2000],
                "min_value": [-125,-125,0,0,-125,0,0],
                "max_value": [124,124,0,0,124,0,0]
},"TM": {"colname": ['bearing','coil'],
                "coltype": ['float','float'],
                "threshold": [100,100],
                "state_num": [500,500],
                "min_value": [-50,-50],
                "max_value": [230,230]
},"AB": {"colname": ['rms','freq_1x','freq_2x','band','BPF1','BPF0','BSF','FTF','Crest','demodul','bearing'],
                "coltype": ['float','float','float','float','float','float','float','float','float','float','float'],
                "threshold": [32,1.6,1.6,12,3.2,3.2,3.2,3.2,64,16,120],
                "state_num": [500,500,500,500,500,500,500,500,500,500,500],
                "min_value": [0,0,0,0,0,0,0,0,0,0,-50],
                "max_value": [100,15,15,50,30,30,30,30,200,100,230]
},"FDT": {"colname": ['temp','dust'],
                "coltype": ['float','float'],
                "threshold": [90,130],
                "state_num": [500,500],
                "min_value": [-50,0],
                "max_value": [230,200]
},"PA": {"colname": ['volt','current','temp','noise'],
                "coltype": ['float','float','float','binary'],
                "threshold": [22,2,95,0],
                "state_num": [500,500,500,500],
                "min_value": [0,0,-50,0],
                "max_value": [25.5,10,125,0]
},"GDB": {"colname": ['relay','circuit'],
                "coltype": ['float','float'],
                "threshold": [90,90],
                "state_num": [500,500],
                "min_value": [0,0],
                "max_value": [100,100]
},"DG": {"colname": ['gear'],
                "coltype": ['float'],
                "threshold": [100],
                "state_num": [500],
                "min_value": [-50],
                "max_value": [230]
}
                
}

with open("D:/Users/Cho/Projects/CBM/data/sensors.json", "w") as json_file:
    json.dump(sensors_json, json_file)
    
#%%

with open("D:/Users/Cho/Projects/CBM/data/sensors.json") as st_json:
    st_python = json.load(st_json)

# %%
st_python["HVAC"]["min_value"]
# %%
