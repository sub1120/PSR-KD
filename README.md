# Automated Shorthand Recognition using Optimized DNNs
This repository contains code used in our paper ***"A Machine-Driven Pitman Shorthand Writing Recognition Model using a Compressed Mobile Network with Knowledge Distillation and Grid-search Optimization"*** (under review). It also has serves as a supplementary for a research article submitted for publication. Please contact us before using any material in this repository.


**Authors:** Subham Sahu, Dr. Francis Jesmar P. Montalbo, Yogesh Kumar Vishwakarma, Jeevanlal Kori.



![Abstract](https://github.com/sub1120/PSR-KD/blob/master/assets/Graphical%20Abstract.png)


## Requirements
- Python 3.7.7
- tensorflow == 2.7.0
- tf-keras-vis == 0.6.2
- matplotlib == 3.5.1
- opencv-python==4.5.5.62
- numpy == 1.21.5
- tqdm == 4.62.3
- scikit-learn == 1.0.2
- Pillow == 9.0.1

Use following command
```pip install -r requirements.txt```

## Directories
- *utils/ :* Defined all Custom functions required for Evaluation of Approach.  
    1. *load_utils.py*: Defined functions to load trained models.
    2. *cam_utils.py*: Defined functions to produce CAMs.
    3. *ensemble.py*: Defined functions to make ensemble model.
    4. *eval_utils.py*: All other required functions are defined here.
- *notebooks :* Contains all Training, Evaluation and Other Notebooks.
- *test.py :* A simple script to test our all models. 
- *gui.py :* GUI Scode.
- *models* : All trained models. [models drive link](https://drive.google.com/drive/folders/16lHHTsGacH6Ov6lxngxjHDl_pKzxfEuZ?usp=sharing)
- *ds* : Dataset  folder. [dataset drive link](https://drive.google.com/drive/folders/1uhZaogn_ksJuppiCPH_0xIGqpLywp8QT?usp=sharing)

## Usage
Below command will open a Interface for testing our models.

> ```python gui.py ``` 

To run script with GPU, use below command
> ```python gui.py -g```

## Interface Guide
![Output](https://github.com/sub1120/PSR-KD/blob/master/out/GUI.PNG)

### Sample Output
|     Model Name      |   Output|
| ------------------- | --------|
| ComMoN (Best Student) | |

**Note1:** CAMERAS will take a while to generate cams.

**Note2:** For NasNetLarge & NasNetMobile use legacy-cameras branch to produce cameras cams.
