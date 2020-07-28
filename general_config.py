import torch
import constants
"""
device - string: cuda:0 or cpu
"""

# training related
amp_opt_level="O2"
use_amp = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
