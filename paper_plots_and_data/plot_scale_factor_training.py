import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from utils.learning_helpers import save_obj, load_obj
import os
import matplotlib
# Removes the XWindows backend (useful for producing plots via tmux without -X)
matplotlib.use('Agg', warn=False)
os.makedirs('figures', exist_ok=True)

path_to_ws = '/home/brandon-wagstaff/learned_scale_recovery/'
scaled_file = path_to_ws+'results/202007111233-kitti-scaled-good/scale_factor'
scale = load_obj(scaled_file)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
f, axarr = plt.subplots(2, sharex=True, sharey=True)
axarr[0].tick_params(labelsize=22)
axarr[1].tick_params(labelsize=22)

axarr[0].plot(scale[1][0:-1500:3], linewidth=2, rasterized=True) 
axarr[1].plot(scale[19][0:-1500:3], linewidth=2, rasterized=True) 
axarr[0].set_ylabel('Scale Factor', fontsize=22)
axarr[1].set_ylabel('Scale Factor', fontsize=22)
axarr[1].set_xlabel('Minibatch', fontsize=22)

axarr[0].set_title('First Epoch', fontsize=19)
axarr[1].set_title('Final Epoch', fontsize=19)
axarr[0].grid()        
axarr[1].grid()    
plt.subplots_adjust(hspace = 0.4)
plt.subplots_adjust(bottom=0.2)

plt.savefig('figures/scale_factor_training.pdf',dpi=400)