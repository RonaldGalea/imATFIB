# imATFIB

Supports ACDC, MMWHS, Imgogen datasets.

Converting to npy:
- it is possible to convert all .nii or .nii.gz files to .npy which is afterwards faster to read during training
Use utils.general.nii_to_npy to convert the dataset, then set read_numpy=True in general_config.py to use this
(downside is .npy are simple binary files so the take more space)

Multi-class vs whole-heart:
MMWHS supports both by default
ACDC only supports multi-class by default
Imogen only support whole-heart by default
