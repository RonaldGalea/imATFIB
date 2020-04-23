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
height, width = 224, 224
matplot = False
exist_label = True
max_plot_nr = 8
evaluation_step = 1
statistics_print_step = 10
visualize_dataset = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
