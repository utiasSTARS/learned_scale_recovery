import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from utils.learning_helpers import save_obj, load_obj
import os
os.makedirs('figures', exist_ok=True)


path_to_ws = '/home/brandon-wagstaff/learned_scale_recovery/'


scaled_file = path_to_ws+'results/202008041237-oxford-to-kitti-1-epoch-scaled/scale_factor'
unscaled_file = path_to_ws+'results/202008041450-oxford-to-kitti-1-epoch-unscaled/scale_factor'

scale = load_obj(scaled_file)
unscale = load_obj(unscaled_file)

scaled_average = []
for i in range(0,len(scale[1]),100):
    scaled_average.append(np.mean(scale[1][i:i+100]))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
f, axarr = plt.subplots(1)
axarr.tick_params(labelsize=19)

axarr.plot(scale[1][0:-1000:3], linewidth=2) #,color='red',linestyle='--')
axarr.plot(unscale[1][0:-1000:3], linewidth=2) #,color='gold',linestyle='-')

axarr.set_ylabel('Scale Factor', fontsize=19)
axarr.set_xlabel('Minibatch', fontsize=19)

axarr.legend(['Scale Recovery Loss', 'Unscaled'], fontsize=14, numpoints=1)

plt.suptitle('Retraining Experiment (Oxford to KITTI)', fontsize=17,y=0.95)
axarr.grid()        

plt.subplots_adjust(bottom=0.2)
plt.savefig('figures/retrained_scale_factor.pdf')