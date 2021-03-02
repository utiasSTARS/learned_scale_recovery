import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from utils.learning_helpers import save_obj, load_obj
import os
# import matplotlib
# Removes the XWindows backend (useful for producing plots via tmux without -X)
# matplotlib.use('Agg', warn=False)


if __name__=='__main__':
    os.makedirs('figures', exist_ok=True)

    path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/'
    scaled_file = path_to_ws+'results/final_models/202102121108-scaled-1-iter-flow-ox_pretrain/scale_factor'
    scale_init = load_obj(scaled_file)

    scaled_file = path_to_ws+'results/final_models/vo-kitti-scaled-202102182020/scale_factor'
    scale_final = load_obj(scaled_file)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # f, axarr = plt.subplots(1,2, sharey=True, sharex=False)
    # axarr[0].tick_params(labelsize=22)
    # axarr[1].tick_params(labelsize=22)

    # axarr[0].plot(scale_init[1][0:-1500:3], linewidth=2, rasterized=True) 
    # axarr[1].plot(scale_final[19][0:-1500:3], linewidth=2, rasterized=True) 
    # axarr[0].set_ylabel('Scale Factor', fontsize=22)
    # # axarr[1].set_ylabel('Scale Factor', fontsize=22)
    # axarr[0].set_xlabel('Minibatch', fontsize=22)
    # axarr[1].set_xlabel('Minibatch', fontsize=22)

    # axarr[0].set_title('First Epoch', fontsize=19)
    # axarr[1].set_title('Final Epoch', fontsize=19)
    # axarr[0].grid()        
    # axarr[1].grid()    
    # # plt.subplots_adjust(hspace = 0.4)
    # plt.subplots_adjust(bottom=0.2)
    # # plt.axis('scaled')
    # plt.savefig('figures/scale_factor_training.pdf',dpi=400)
    
    fig = plt.figure()
    plt.tight_layout()
    ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=600)
    ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=5*470)
    
    
    ax1.tick_params(labelsize=22)
    ax2.tick_params(labelsize=22)

    ax1.plot(scale_init[1][0:-1500:3], linewidth=2, rasterized=True) 
    ax2.plot(scale_final[19][0:-1500:3], linewidth=2, rasterized=True) 
    ax1.set_ylabel('Scale Factor', fontsize=22)
    # ax2.set_ylabel('Scale Factor', fontsize=22)
    ax1.set_xlabel('Minibatch', fontsize=22)
    ax2.set_xlabel('Minibatch', fontsize=22)

    ax1.set_ylim([0.8,2.2])
    ax2.set_ylim([0.8,2.2])

    ax1.set_title('First Epoch', fontsize=19)
    ax2.set_title('Final Epoch', fontsize=19)
    ax1.grid('both')        
    ax2.grid('both')    
    # plt.subplots_adjust(hspace = 0.4)
    # plt.subplots_adjust(bottom=0.2)
    # plt.axis('scaled')
    plt.savefig('figures/scale_factor_training.pdf',dpi=400,bbox_inches='tight')
        