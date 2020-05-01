# imATFIB

Supports ACDC, MMWHS, Imogen datasets.

Data loading:
- since the datasets are small enough to fit in memory, they are loaded and just indexed when training is done.

Data preprocessing:
- normalization: support per slice, per volume or over the whole dataset, also will look more into dealing with outliers.
- augmentation: horizontal flips, rotations, elastic distorsions, varying resolutions. One other important factor is cropping, since labels usually are located in a fairly small region compared to the whole image, cropping can be quite efficient.
- loading 2D slices: it somewhat problematic to have a fixed size batch parameter when working with volumes (as they have different depths), that is why for now I simply break everything up in a long list of 2D slices and construct batches from those.

Training:
- loss function: standard pixel wise cross entropy loss
- models: standard Unet and DeepLabV3+ with MobileNetV2

Evaluation:
- for the evaluation to be fair, the validation volumes are loaded without being modified and instead the model predictions are upsampled (immediately after applying logsoftmax, to take advantage of upsampling on a continuous domain) to the original resolution. Then the average dice is the mean of all the volumes' dices. 
- Using official ACDC metrics (https://www.creatis.insa-lyon.fr/Challenge/acdc/code/metrics_acdc.py)

Multi-class vs whole-heart:
MMWHS supports both by default
ACDC only supports multi-class by default
Imogen only support whole-heart by default
