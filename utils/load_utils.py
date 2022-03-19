#Import models API
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_resnet_v2 
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import resnet_v2 
from keras.applications import efficientnet
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

#List all model names
BASE_FILES = ['DenseNet201', 'EfficientNetB7', 'InceptionResNetV2', 'ResNet50V2', 'ResNet152V2', 'NASNetLarge', 'Xception', 'InceptionV3', 'DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'EnsembleModel']
PROPOSED_FILES = ['KD-ComMoN', 'ComMoN']
MODEL_FILES = PROPOSED_FILES + BASE_FILES

#Load models function
def init_models(model_path, model_name):
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
	models['ComMoN']['preprocess_func'] = mobilenet_v2.preprocess_input
	models['KD-ComMoN']['preprocess_func'] = mobilenet_v2.preprocess_input

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
	models['ComMoN']['image_size'] = (224, 224)
	models['KD-ComMoN']['image_size'] = (224, 224)
	models['EnsembleModel']['image_size'] = (224, 224)

	model = None
	#Load required models

	print("[INFO] Loading Model: " + model_name)
	if model_name != 'EnsembleModel':
		model = load_m(model_path + model_name, model_name)
	else:
		custom_layers = {'EN_PreprocessLayer':EN_PreprocessLayer, 
		                 'Dense_PreprocessLayer':Dense_PreprocessLayer,
		                 'NN_PreprocessLayer':NN_PreprocessLayer,
		                 'INV3_PreprocessLayer':INV3_PreprocessLayer,
		       			}
		model = load_m(model_path + model_name, model_name, custom_objects=custom_layers)

	models[model_name]['model'] = model

	return models

#Load model Function
def load_m(directory, model_name, custom_objects=None):
    if not os.path.exists(directory):
        print("Model File Does Not Exist!!")
        exit() 
    model = load_model(directory + "/" + model_name + ".h5", custom_objects=custom_objects)
    return model

 #DATA GENERATORS
def create_data_generator(data_path, input_shape=(224,224), batch_size=4, pre_process=None):
    print("INPUT SIZE -->", input_shape)
    print("BATCH SIZE -->", batch_size)

    nb_samples = 0
    generator = None

    datagen = ImageDataGenerator(preprocessing_function=pre_process)
    
    if not os.path.exists(data_path):
        print("DATA DOES NOT EXITS!")
    else:
        print("LOADING SAMPLES from ", data_path, "...")
        generator = datagen.flow_from_directory(
                data_path,
                target_size=input_shape,
                batch_size=batch_size,
                class_mode='categorical',
                seed=42,
                shuffle=False)

        #CHECK  THE NUMBER OF SAMPLES
        nb_samples = len(generator.filenames)
        if nb_samples == 0:
            print("NO DATA please check the path! ", data_path)

    return generator