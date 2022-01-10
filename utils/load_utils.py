#Import models API
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_resnet_v2 
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import resnet_v2 
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import mobilenet_v2

#Import ensemble modules
from utils.ensemble import EN_PreprocessLayer
from utils.ensemble import Dense_PreprocessLayer
from utils.ensemble import NN_PreprocessLayer
from utils.ensemble import INV3_PreprocessLayer

#Import other modules 
import numpy as np
import pickle
import math
import os

#Define model paths
BASE_PATH = 'models/teacher_model'
PROPOSED_PATH = 'models/proposed_model'

#List all model names
BASE_FILES = ['DenseNet201', 'EfficientNetB7', 'InceptionResNetV2', 'ResNet50V2', 'ResNet152V2', 'NASNetLarge', 'Xception', 'InceptionV3', 'DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'EnsembleModel', 'MiniMobileNetV2']
PROPOSED_FILES = ['MiniMobileNetV2', 'MiniMobileNetV2-KD']
MODEL_FILES = BASE_FILES + PROPOSED_FILES

#Load models function
def init_models(load_files):
	#Initialize models
	models = {}
	for file in MODEL_FILES:
		models[file] = {}
		models[file]['preprocess_func'] = None
		models[file]['image_size'] = None
		models[file]['model'] = None		
		
	#Set Preprocessing Functions of models
	models['DenseNet201']['preprocess_func'] = densenet.preprocess_input
	models['EfficientNetB7']['preprocess_func'] = efficientnet.preprocess_input
	models['InceptionResNetV2']['preprocess_func'] = inception_resnet_v2.preprocess_input
	models['ResNet152V2']['preprocess_func'] = resnet_v2.preprocess_input
	models['NASNetLarge']['preprocess_func'] = nasnet.preprocess_input
	models['Xception']['preprocess_func'] = xception.preprocess_input
	models['InceptionV3']['preprocess_func'] = inception_v3.preprocess_input
	models['DenseNet121']['preprocess_func'] = densenet.preprocess_input
	models['EfficientNetB0']['preprocess_func'] = efficientnet.preprocess_input
	models['ResNet50V2']['preprocess_func'] = resnet_v2.preprocess_input
	models['MobileNetV2']['preprocess_func'] = mobilenet_v2.preprocess_input
	models['NASNetMobile']['preprocess_func'] = nasnet.preprocess_input
	models['MiniMobileNetV2']['preprocess_func'] = mobilenet_v2.preprocess_input
	models['MiniMobileNetV2-KD']['preprocess_func'] = mobilenet_v2.preprocess_input

	#Set input size of models
	models['DenseNet201']['image_size'] = (224, 224)
	models['EfficientNetB7']['image_size'] = (224, 224)
	models['InceptionResNetV2']['image_size'] = (299, 299)
	models['ResNet152V2']['image_size'] = (224, 224)
	models['NASNetLarge']['image_size'] = (331, 331)
	models['Xception']['image_size'] = (299, 299)
	models['InceptionV3']['image_size'] = (299, 299)
	models['DenseNet121']['image_size'] = (224, 224)
	models['EfficientNetB0']['image_size'] = (224, 224)
	models['ResNet50V2']['image_size'] = (224, 224)
	models['MobileNetV2']['image_size'] = (224, 224)
	models['NASNetMobile']['image_size'] = (224, 224)
	models['MiniMobileNetV2']['image_size'] = (224, 224)
	models['MiniMobileNetV2-KD']['image_size'] = (224, 224)
	models['EnsembleModel']['image_size'] = (224, 224)

	model = None
	#Load required models
	for i in range(len(load_files)):
		print("[INFO] Loading Model: " + load_files[i])
		file = load_files[i]
		#If model is teacher
		if file in BASE_FILES:
			if file != 'EnsembleModel':
				model = load_m(BASE_PATH + '/' + file, file)
			else:
				custom_layers = {'EN_PreprocessLayer':EN_PreprocessLayer, 
				                 'Dense_PreprocessLayer':Dense_PreprocessLayer,
				                 'NN_PreprocessLayer':NN_PreprocessLayer,
				                 'INV3_PreprocessLayer':INV3_PreprocessLayer,
				       			}
				model = load_m(BASE_PATH + '/' + file, file, custom_objects=custom_layers)
		#If model is normal student
		else:
			model = load_m(PROPOSED_PATH + '/' + file, file)
				
		models[file]['model'] = model

	return models

#Load model Function
def load_m(directory, model_name, custom_objects=None):
    if not os.path.exists(directory):
        print("Model File Does Not Exist!!")
        exit() 
    model = load_model(directory + "/" + model_name + ".h5", custom_objects=custom_objects)
    return model