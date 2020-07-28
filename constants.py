# directory names and other string constants used throughout the poject

acdc_root_dir = "ACDC_training"
imatfib_root_dir = "imatfib-whs"
mmwhs_root_dir = "mmwhs"

multi_class_seg = "multiple-classes"
whole_heart_seg = "whole-heart"

per_slice = "per_slice"
per_dataset = "per_dataset"
norm_types = [per_slice, per_dataset]

config_path = "experiments/{}/{}/config.json"
params_path = "experiments/{}/{}/params.json"
stats_path = "experiments/{}/{}/stats.json"
model_path = "experiments/{}/{}/model_checkpoint.pt"

unet = "2D_Unet"
deeplab = "DeepLabV3_plus"
resnext_deeplab = "ResNeXt_DeepLabV3_plus"
segmentor_ids = [unet, deeplab, resnext_deeplab]

resnet18_detector = "resnet18_detector"
resnet50_detector = "resnet50_detector"
detectors = [resnet18_detector, resnet50_detector]

acdc_heart = ["LVC", "LVMyo", "RVC"]
mmwhs_heart = ["LVC", "LVMyo", "RVC", "LA", "RA", "AA", "PA"]

divide_decay = "divide"
poly_decay = "poly"
lr_schedulers = [divide_decay, poly_decay]

optimizers = ["adam"]

no_roi_extraction = "no_roi"
relative_roi_extraction = "relative_roi"
roi_types = [no_roi_extraction, relative_roi_extraction]

no_augmentation = "no_aug"
simple_augmentation = "simple"
heavy_augmentation = "heavy"
aug_types = [no_augmentation, simple_augmentation, heavy_augmentation]

results_overlay_gt = "gt_over"
results_overlay_inp = "inp_over"

load_simple = "load_simple"
load_transfer = "transfer_learning"
load_types = [load_simple, load_transfer]

classifier_freeze = "classifier_layer"
progressive_freeze = "all_layers"
freeze_types = [classifier_freeze, progressive_freeze]
