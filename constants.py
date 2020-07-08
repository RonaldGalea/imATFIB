# directory names and other string constants used throughout the poject

acdc_root_dir = "ACDC_training"
imatfib_root_dir = "imatfib-whs"
mmwhs_root_dir = "mmwhs"

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
resnext_deeplab = "ResNeXt_DeepLabV3_plus"
model_ids = [unet, deeplab, resnext_deeplab]

acdc_heart = ["LVC", "LVMyo", "RVC"]
mmwhs_heart = ["LVC", "LVMyo", "RVC", "LA", "RA", "AA", "PA"]

divide_decay = "divide"
poly_decay = "poly"
lr_schedulers = [divide_decay, poly_decay]

no_roi_extraction = "no_roi"
relative_roi_extraction = "relative_roi"
global_roi_extraction = "global_roi"
roi_types = [no_roi_extraction, relative_roi_extraction, global_roi_extraction]

no_augmentation = "no_aug"
simple_augmentation = "simple"
heavy_augmentation = "heavy"
aug_types = [no_augmentation, simple_augmentation, heavy_augmentation]

results_overlay_gt = "gt_over"
results_overlay_inp = "inp_over"

no_load = "none"
load_training = "resume_training"
load_transfer = "transfer_learning"
load_types = [load_training, load_transfer, no_load]

no_freeze = "none"
classifier_freeze = "classifier_layer"
progressive_freeze = "all_layers"
freeze_types = [no_freeze, classifier_freeze, progressive_freeze]
