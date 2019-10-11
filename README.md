# Team: Expasoft
> ImageNet classification

# Requirements
  * Python: 3.6.8
  * numpy: 1.16.0 - 1.16.4
  * tensorflow-gpu: 1.12.2
  * click: 7.0

# Results validation

 1. Download this repo
 2. Download a [ready-to-use model data](https://yadi.sk/d/0ri9Qk1GJdNomA) with trained weights (21.4 MB)
 3. Put files **'ready_to_use_thresholds_vector.pickle'** and **'ready_to_use_weights.pickle'** into `model-data` directory
 4. Download the [ImageNet validation dataset packed into a TF-Records file](https://yadi.sk/d/UVI9JniKaspHEg) (6 GB)
 5. Put the TF-Record file or a symlink to it to the `dataset-data` directory (the script expects to see a file `dataset-data/val_set`).
 6. Run the script `evaluate_ready_to_use_model.py`. The Top-1 is about 75.6%.
 7. Run the script `evaluate_score.py`.
 
> For validation we are using integer arithmetic.
> In most cases integer numbers are stored in float32 containers 
> (Capabale of representing integers values up to 2 ** 23).
> For Convoluton and Matrix multiplication we are using float64.
> Each requantization is followed by clipping.
> Each quantization is followed by rounding and clipping.
 
 ### Results
  * Top-1: 75.60% (image size 224x224)
  * Score: 0.88 / 6.9 + 315.9 / 1170 = 0.3974


# Reproducing the Results

 1. Download this repo
 2. Download original [EfficientNet-B0 model](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz) (37.5 MB)
 3. Download the [ImageNet validation dataset packed into a TF-Records file](https://yadi.sk/d/UVI9JniKaspHEg) (6 GB)
 4. Put the TF-Record file or a symlink to the `dataset-data` directory (the script expects to see a file `dataset-data/val_set`).
 5. Download the [ImageNet train subset](https://drive.google.com/drive/folders/1xEH1DejM2e7sj1cn69H5S6rIASiVdtZO?usp=sharing) (12 GB)
 6. Put the two TF-Record files or symlinks to them to the `dataset-data` directory 
    (the script expects to see two files `dataset-data/train_set0000` and `dataset-data/train_set0001`).
 7. Run the script `extract_weights_from_the_original_model.py` with 
    specifying the path to the original pb-file  `--pb-path`
 8. Run the script `quantize_model.py` (With GeForce 1080 Ti the process takes about 40 minutes).
 9. Run the script `evaluate_model.py` (2-3 hours)
