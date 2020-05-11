import torch

"""
exist_label - bool: true will show labels regardless if there's any, false will only show
images where the heart is visible
height, width - int: displayed height and width, only used in visualization (to change training
resolution look in the .json parameter file of the respective experiment)
matplot - bool: use matplotlib to plot (better colors imo)
max_plot_nr - int: maximum number in matplot
read_numpy - bool: obsolete
device - string: cuda:0 or cpu
evaluation_step - int: after how many epochs to evaluate
statistics_print_step - int: print statistics every len(dataset) / statistics_print_step samples
visualize_dataset - bool: visually inspect image/mask before training starts
"""


read_numpy = False
height, width = 400, 400
matplot = False
exist_label = True
max_plot_nr = 8
evaluation_step = 5
statistics_print_step = 3
visualize_dataset = False
use_amp = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_mean = 56.878749106586845
dataset_std = 139.08557576261646
seg_type = "whole-heart"
# device = "cpu"


"""
Imatfib dset mean and std
56.878749106586845 139.08557576261646
"""
