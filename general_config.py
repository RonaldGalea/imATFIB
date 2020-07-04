import torch
import constants

"""
exist_label - bool: true will show labels regardless if there's any, false will only show
images where the heart is visible
height, width - int: displayed height and width, only used in visualization (to change training
resolution look in the .json parameter file of the respective experiment)
matplot - bool: use matplotlib to plot (better colors imo)
max_plot_nr - int: maximum number in matplot
device - string: cuda:0 or cpu
evaluation_step - int: after how many epochs to evaluate
statistics_print_step - int: print statistics every len(dataset) / statistics_print_step samples
visualize_dataset - bool: visually inspect image/mask before training starts
"""

# non training
height, width = 400, 400
matplot = False
exist_label = True
max_plot_nr = 8
evaluation_step = 10
statistics_print_step = 3
visualize_dataset = False

# training related
use_amp = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seg_type = constants.whole_heart_seg
