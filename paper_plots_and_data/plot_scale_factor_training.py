import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from utils.learning_helpers import save_obj, load_obj

import matplotlib
# Removes the XWindows backend (useful for producing plots via tmux without -X)
matplotlib.use('Agg', warn=False)

###Moving average filter of size W
def moving_average(a, n) : #n must be odd)
    if n == 1:
        return a
    else:
        for i in range(a.shape[1]):
            if (n % 2) == 0:
                n -=1
            ret = np.cumsum(a[:,i], dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            a[:,i] = np.pad(ret[n - 1:-2] / n , int((n-1)/2+1), 'edge')
        return a

scaled_file = '/home/brandon-wagstaff/learned_scale_recovery/results/202007111233-kitti-scaled-good/scale_factor'
scale = load_obj(scaled_file)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
f, axarr = plt.subplots(2, sharex=True, sharey=True)
axarr[0].tick_params(labelsize=22)
axarr[1].tick_params(labelsize=22)



axarr[0].plot(scale[1][0:-1500:3], linewidth=2, rasterized=True) #,color='gold',linestyle='-')

axarr[1].plot(scale[19][0:-1500:3], linewidth=2, rasterized=True) #,color='gold',linestyle='-')

axarr[0].set_ylabel('Scale Factor', fontsize=22)
axarr[1].set_ylabel('Scale Factor', fontsize=22)
# axarr[0].set_xlabel('Minibatch', fontsize=19)
axarr[1].set_xlabel('Minibatch', fontsize=22)


# axarr[0].legend(['Unscaled', 'Scaled (Ours)'], fontsize=14, numpoints=1)
# axarr[1].legend(['Unscaled', 'Scaled (Ours)'], fontsize=14, numpoints=1)

axarr[0].set_title('First Epoch', fontsize=19)
axarr[1].set_title('Final Epoch', fontsize=19)
axarr[0].grid()        
axarr[1].grid()    
plt.subplots_adjust(hspace = 0.4)
plt.subplots_adjust(bottom=0.2)
# axarr[0].set_xlim([0.1,4.5])
# axarr[1].set_xlim([1e7,9e8])    
# axarr[0].set_ylim([0,2])  
# axarr[1].set_ylim([0,2])  
##axarr[0].yaxis.set_ticks(np.arange(0,2,0.2))
##axarr[1].yaxis.set_ticks(np.arange(0,2,0.2))    
# plt.savefig('thresh-error-SHOE.eps', format='eps', dpi=800, bbox_inches='tight')

plt.savefig('figures/scale_factor_training.pdf',dpi=400)