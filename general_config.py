import torch

read_numpy = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
