import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd

# Pretty Print model names
auto_names = {"BW535i90":"BMW 535i",
"BW735i88":"BMW 735i",
"CDDEVI90":"Cadillac DeVille",
"CVCORS87":"Chevy Corsica",
"FDTHND90":"Ford Thunderbird",
"HDACCO90":"Honda Accord",
"MB420S87":"Mercedes Benz 420s",
"TYCAMR87":"Toyota Camry"}

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

def nice_name(lookup) :
    try:
        return auto_names[lookup]
    except:
        return lookup


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
