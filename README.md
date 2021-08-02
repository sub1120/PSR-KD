<!-- # Automated Shorthand Recognition using Optimized DNNs
This repository contains code used in our paper ***"Automated Shorthand Writing Recognition using an Optimized Knowledge Distilled Fine-Tuned Deep Convolutional Neural Network"*** (under review). -->

**Authors:** Subham Sahu, Dr. Francis Jesmar P. Montalbo, Yogesh Kumar Vishwakarma, Jeevanlal Kori.

## Requirements
- Python 3.7.7
- tensorflow 2.5.0
- tf-keras-vis 0.6.2
- keras-flops 0.1.2
- matplotlib 3.4.2
- opencv-python 4.5.2.54
- numpy 1.19.5
- tqdm 4.61.1

Use following command
```pip install -r requirements.txt```

## Directories
- *utils/ :* Defined all Custom functions required for Evaluation of Approach.  
    1. *load_utils.py*: Defined functions to load trained models.
    2. *cam_utils.py*: Defined functions to produce CAMs.
    3. *plot_utils.py*: Defined functions to draw evaluation plots.
    4. *ensemble.py*: Defined functions to make ensemble model.
    5. *eval_utils.py*: All other required functions are defined here.
- *notebooks :* Contains all Training, Evaluation and Other Notebooks.
- *test.py :* A simple script to try our all models, usage is defined below. 


## Usage
```python test.py [-h] [-o OUTPUT_PATH] [-m MODEL_INDEX] [-v] [-c] [-p] [-s] [-k] [-g] ```

optional arguments:
```
                             DESCRIPTION                  DEFAULT VALUE
-----------------------------------------------------------------------------
   -o     --output_path     output directory                   out
   -m     --model_index     model indexes                       1
   -v     --video_mode      enable video mode                 False
   -c     --gradcam         obtain grad cam                   False
   -p     --gradcamplus     obtain grad cam++                 False
   -s     --scorecam        obtain faster score cam           False
   -k     --camerascam      obtain CAMERAS Cam                False
   -g     --enable_gpu      enable gpu                        False
 ```
 
- **Usage 1:** Use *VIDEO MODE* for prediction.

> ```python test.py --model_index 1 --video_mode```
 
- **Usage 2:** Use *VIDEO MODE* for prediction with Cam Output.

> ```python test.py --model_index 1 --video_mode --gradcamplus```

- **Usage 3:** Use *IMAGE MODE* for prediction.

> ```python test.py --input_path sample_images --output_path out --model_index 1```

- **Usage 4:** Use *IMAGE MODE* for prediction with Cams Output.

> ```python test.py --input_path sample_images --output_path out --model_index 1 --gradcam --gradcamplus```

**Note:** In *VIDEO MODE* enable only one cam type, but in *IMAGE MODE* you can enable multiple Cams.
 
***Use below Model Indexes***

|     KD-Student      | Index   |    Student         |  Index   |    Teacher         | Index   |
| ------------------- | --------| -------------------| --------| ------------------- | --------|
|    KD-DenseNet121   |    0    | DenseNet121        |    5    | DenseNet201         |   10    |
|  KD-EfficientNetB0  |    1    |EfficientNetB0      |    6    |   Xception          |   11    |
|   KD-NASNetMobile   |    2    |NASNetMobile        |    7    |  InceptionResNetV2  |   12    |
|   KD-MobileNetV2    |    3    | MobileNetV2        |    8    |  ResNet152V2        |   13    |
| KD-Custom-CNN       |    4    | Custom-CNN         |    9    |  EfficientNetB7     |   14    |
||||                                                           |    NASNetLarge      |   15    |
||||                                                           |    EnsembleModel    |   16    |

### Sample Output from Usage 3
