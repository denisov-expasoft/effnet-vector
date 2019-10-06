# Team: Expasoft

# Requirements
  * Python: 3.6.8
  * numpy: 1.16.0 - 1.16.4
  * tensorflow-gpu: 1.12.2
  * opencv-python: 3.4.1.15
  * dynaconf: 2.0.1
  * click: 7.0

# Results validation

 1. Download this repo
 2. Download a [ready-to-use tflite model](https://yadi.sk/d/A0OJB5ovvg-aPQ) with trained weights (6 MB)
 3. Put the tflite model into `model-data` directory
 4. Download the [ImageNet validation dataset packed into a TF-Records file](https://yadi.sk/d/UVI9JniKaspHEg) (6 GB)
 5. Put the TF-Record file or a symlink to it to the `dataset-data` directory (the script expects to see a file `dataset-data/val_set`).
 6. Run the script `evaluate_ready_to_use_model.py`. The Top-1 is about 75.8%.
    (Depending on a CPU, the process can take from 2 to 3 hours)
 7. Run the script `evaluate_score.py`.
 
> For validation we are using TFLite Interpretor with 8-bit quantization.
 
 ### Results
  * Top-1: 75.83%
  * Score: 1.55 / 6.9 + 643.2 / 1170 = 0.7751


# Reproducing the Results

 1. Download this repo
 2. Download original [Mnasnet-1.3-224 model](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.3_224_09_07_2018.tgz) (24 MB)
 3. Download the [ImageNet validation dataset packed into a TF-Records file](https://yadi.sk/d/UVI9JniKaspHEg) (6 GB)
 4. Put the TF-Record file or a symlink to the `dataset-data` directory (the script expects to see a file `dataset-data/val_set`).
 5. Download the [ImageNet train subset](https://drive.google.com/drive/folders/1xEH1DejM2e7sj1cn69H5S6rIASiVdtZO?usp=sharing) (12 GB)
 6. Put the two TF-Record files or symlinks to them to the `dataset-data` directory 
    (the script expects to see two files `dataset-data/train_set0000` and `dataset-data/train_set0001`).
 7. Run the script `extract_weights_from_the_original_model.py` with 
    specifying the path to the original pb-file  `--pb-path`
 8. Run the script `quantize_model.py` (With GeForce 1080 Ti the process takes about 40 minutes).
 9. Run the script `evaluate_model.py` (2-3 hours)
