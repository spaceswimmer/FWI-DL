import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def plot_3d_slices(array_3d, seismic):
    """
    Plot 3d array right next to seismic. Also has a slider for choosing the inline
    """
    fig, axs = plt.subplots(1,2)
    plt.subplots_adjust(bottom=0.2)
    
    # Initial slice
    initial_slice = 0
    im1 = axs[0].imshow(array_3d[:, initial_slice, :], cmap='viridis', aspect='auto')
    axs[0].set_title(f'Slice {initial_slice}')
    im2 = ax.imshow(seismic[:, initial_slice, :], cmap='seismic', aspect='auto')
    ax.set_title(f'Slice {initial_slice}')

    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, array_3d.shape[1] - 1, valinit=0, valfmt='%d')
    
    def update(val):
        slice_num = int(slider.val)
        im1.set_array(array_3d[:, slice_num, :])
        axs[0].set_title(f'Slice {slice_num}')
        im2.set_array(seismic[:, slice_num, :])
        ax.set_title(f'Slice {slice_num}')
        fig.canvas.draw()
    
    slider.on_changed(update)
    plt.show()

def plot_3d_seismic(seismic):
    """
    Plot 3d array right next to seismic. Also has a slider for choosing the inline
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    
    # Initial slice
    initial_slice = 0
    im2 = ax.imshow(seismic[:, initial_slice, :], cmap='seismic', aspect='auto')
    ax.set_title(f'Slice {initial_slice}')

    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, seismic.shape[1] - 1, valinit=0, valfmt='%d')
    
    def update(val):
        slice_num = int(slider.val)
        im2.set_array(seismic[:, slice_num, :])
        ax.set_title(f'Slice {slice_num}')
        fig.canvas.draw()
    
    slider.on_changed(update)
    plt.show()