import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd

import time
_start_time = time.time()
def tick():
    global _start_time 
    _start_time = time.time()
def tock():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
    
    
    
    
# File Directories
#
## Plot Configuration: Prior to June 2020
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.style.use('fivethirtyeight')
#matplotlib.rcParams.update({'font.size': 24})
#plt.rc('font', size=24)          # controls default text sizes
#plt.rc('axes', titlesize=24)     # fontsize of the axes title
#plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
#plt.rc('legend', fontsize=24)    # legend fontsize
#plt.rc('figure', titlesize=24)
#
#

def nice_name(lookup) :
    tab_name = ["BW535i90", "BW735i88",
                "CDDEVI90", "CVCORS87",
                "FDTHND90", "HDACCO90",
                "MB420S87", "TYCAMR87"
                ]
    tab_nice_name = [
            "BMW 535i",
            "BMW 735i",
            "Cadillac DeVille",
            "Chevy Corsica",
            "Ford Thunderbird",
            "Honda Accord",
            "Mercedes Benz 420s",
            "Toyota Camry"
            ]
    
    tab_nn = pd.DataFrame([tab_name, tab_nice_name]).transpose()
    tab_nn.columns = ['name', 'nice_name']
    
    #tab_names.lookup(tab_names['name'])
    tmp = tab_nn.loc[tab_nn['name'] == lookup].reset_index()
    if len(tmp) == 0:
        return lookup
    val = tmp['nice_name'][0]
    return val


# Plot Configuration: Used as of June 2020
matplotlib.style.use('seaborn-whitegrid')
matplotlib.rcParams.update({'font.size': 24})

plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=24)
#plt.rc('axes',prop_cycle=cycler(color=['#252525', '#636363', '#969696', '#bdbdbd'])*cycler(linestyle=['-',':','--', '-.']))
plt.rc('lines', linewidth=3)
