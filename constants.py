# directory names and other string constants used throughout the poject

acdc_root_dir = "ACDC_training"
imatfib_root_dir = "imatfib-whs"
mmwhs_root_dir = "mmwhs"

acdc_root_dir_npy = "ACDC_training_npy"
imatfib_root_dir_npy = "imatfib-whs_npy"
mmwhs_root_dir_npy = "mmwhs_npy"

multi_class_seg = "multiple-classes"
whole_heart_seg = "whole-heart"

per_slice = "per_slice"
per_dataset = "per_dataset"
norm_types = [per_slice, per_dataset]

params_path = "experiments/{}/{}/params.json"
stats_path = "experiments/{}/{}/stats.json"
model_path = "experiments/{}/{}/model_checkpoint.pt"

unet = "2D_Unet"
deeplab = "DeepLabV3_plus"

heart = ["LV", "RV", "MYO"]
