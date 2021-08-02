#Import layers and model utils
from tensorflow.keras.layers import Dropout, Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model, load_model

#Import models API
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_resnet_v2 
from tensorflow.keras.applications import resnet_v2 
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import mobilenet_v2

#Import ensemble modules
from utils.ensemble import ENB7_PreprocessLayer
from utils.ensemble import D201_PreprocessLayer
from utils.ensemble import XCEP_PreprocessLayer
from utils.ensemble import IRNV2_PreprocessLayer
from utils.ensemble import RNV2_PreprocessLayer
from utils.ensemble import NNL_PreprocessLayer

#Import other modules 
import numpy as np
import pickle
import math
import os

#Define Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
TOTAL_CLASSES = 199

#Define model paths
DIST_STUDENT_MODEL_PATH = 'models/Distilled_Student_Models/'
NORMAL_STUDENT_MODEL_PATH = 'models/Student_Models/'
TEACHER_MODEL_PATH = 'models/Teacher_Models/'

#List all model names
DISTILLED_STUDENT_FILES = ['KD-DenseNet121', 'KD-EfficientNetB0', 'KD-NASNetMobile', 'KD-MobileNetV2', 'KD-Custom-CNN']
NORMAL_STUDENT_FILES = ['DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'Custom-CNN']
TEACHER_FILES = ['DenseNet201', 'Xception', 'InceptionResNetV2', 'ResNet152V2', 'EfficientNetB7', 'NASNetLarge', 'EnsembleModel']
MODEL_FILES = DISTILLED_STUDENT_FILES + NORMAL_STUDENT_FILES +  TEACHER_FILES

#Load models function
def init_models(load_files, image_size=IMAGE_SIZE):
	#Initialize models
	models = {}
	for file in MODEL_FILES:
		models[file] = {}
		models[file]['preprocess_func'] = None
		models[file]['preprocess_layer'] = None
		models[file]['api_func'] = None
		models[file]['rescale'] = None
		models[file]['color'] = None
		models[file]['model'] = None		
		
	#Set Preprocessing Functions of models
	models['DenseNet201']['preprocess_func'] = densenet.preprocess_input
	models['Xception']['preprocess_func'] = xception.preprocess_input
	models['InceptionResNetV2']['preprocess_func'] = inception_resnet_v2.preprocess_input
	models['ResNet152V2']['preprocess_func'] = resnet_v2.preprocess_input
	models['EfficientNetB7']['preprocess_func'] = efficientnet.preprocess_input
	models['NASNetLarge']['preprocess_func'] = nasnet.preprocess_input

	models['EfficientNetB0']['preprocess_func'] = efficientnet.preprocess_input
	models['DenseNet121']['preprocess_func'] = densenet.preprocess_input
	models['NASNetMobile']['preprocess_func'] = nasnet.preprocess_input
	models['MobileNetV2']['preprocess_func'] = mobilenet_v2.preprocess_input

	models['KD-EfficientNetB0']['preprocess_func'] = efficientnet.preprocess_input
	models['KD-DenseNet121']['preprocess_func'] = densenet.preprocess_input
	models['KD-NASNetMobile']['preprocess_func'] = nasnet.preprocess_input
	models['KD-MobileNetV2']['preprocess_func'] = mobilenet_v2.preprocess_input

	#Set Preprocess layers of models(Used only in Ensemble)
	models['DenseNet201']['preprocess_layer'] = D201_PreprocessLayer
	models['Xception']['preprocess_layer'] = XCEP_PreprocessLayer
	models['InceptionResNetV2']['preprocess_layer'] = IRNV2_PreprocessLayer
	models['ResNet152V2']['preprocess_layer'] = RNV2_PreprocessLayer
	models['EfficientNetB7']['preprocess_layer'] = ENB7_PreprocessLayer
	models['NASNetLarge']['preprocess_layer'] = NNL_PreprocessLayer

	#Set Rescale of models(Used only in Custom CNN)
	models['Custom-CNN']['rescale'] = 1./255
	models['KD-Custom-CNN']['rescale'] = 1./255

	#Set Keras API Function of models
	models['DenseNet201']['api_func'] = densenet.DenseNet201
	models['Xception']['api_func'] = xception.Xception
	models['InceptionResNetV2']['api_func'] = inception_resnet_v2.InceptionResNetV2
	models['ResNet152V2']['api_func'] = resnet_v2.ResNet152V2
	models['EfficientNetB7']['api_func'] = efficientnet.EfficientNetB7
	models['NASNetLarge']['api_func'] = nasnet.NASNetLarge

	models['EfficientNetB0']['api_func'] = efficientnet.EfficientNetB0
	models['DenseNet121']['api_func'] = densenet.DenseNet121
	models['NASNetMobile']['api_func'] = nasnet.NASNetMobile
	models['MobileNetV2']['api_func'] = mobilenet_v2.MobileNetV2

	models['KD-EfficientNetB0']['api_func'] = efficientnet.EfficientNetB0
	models['KD-DenseNet121']['api_func'] = densenet.DenseNet121
	models['KD-NASNetMobile']['api_func'] = nasnet.NASNetMobile
	models['KD-MobileNetV2']['api_func'] = mobilenet_v2.MobileNetV2

	#Set colors for models
	i=0
	colors = ['chocolate','olive', 'c','m', 'royalblue' , 									#KD-Student colors
	'chocolate','olive', 'c','m', 'royalblue', 			 									#Normal Student colors
	'tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:brown', 'navy'		#Teacher colors
	]
	for file in MODEL_FILES:
		models[file]['color'] = colors[i]
		i += 1

	model = None
	#Load required models
	for i in range(len(load_files)):
		print("[INFO] Loading Model: " + load_files[i])
		file = load_files[i]
		#If model is teacher
		if file in TEACHER_FILES:
			if file != 'EnsembleModel':
				model = load_m(TEACHER_MODEL_PATH + file)
				model = convert_to_functional(model, models[file]['api_func'])
			else:
				custom_objects = {'ENB7_PreprocessLayer':ENB7_PreprocessLayer,
								  'D201_PreprocessLayer':D201_PreprocessLayer}
				model = load_model(TEACHER_MODEL_PATH + file +'/'+ 'model.h5', custom_objects=custom_objects, compile=False)
		#If model is normal student
		elif file in NORMAL_STUDENT_FILES:
			if file != 'Custom-CNN':
				model = load_m(NORMAL_STUDENT_MODEL_PATH + file)
				model = convert_to_functional(model, models[file]['api_func'])
			else:
				model = load_m(NORMAL_STUDENT_MODEL_PATH + file)
				
		#If model is KD student
		elif file in DISTILLED_STUDENT_FILES:
			if file != 'KD-Custom-CNN':	
				model = load_m(DIST_STUDENT_MODEL_PATH + file)
				model = convert_to_functional(model, models[file]['api_func'])
			else:
				model = load_m(DIST_STUDENT_MODEL_PATH + file)
				
		models[file]['model'] = model

	return models

#Load Model Function
def load_m(file):
	if not os.path.exists(file) and file != 'EnsembleModel':
		print("\n[ERROR] Model path '" + file +  "' does not exist.")
		exit()

	with open(file + "/model.json", "r") as json_file:
		model = json_file.read()
		model = model_from_json(model)
		model.load_weights(file+'/model.h5')
		return model

#Convert Sequential model to Functional model of keras
def convert_to_functional(model, api_func):
	transfer = api_func(include_top = False, weights=None, input_tensor=Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
	x = GlobalAveragePooling2D()(transfer.layers[-1].output)
	x = Dropout(0.5)(x)
	outputs = Dense(TOTAL_CLASSES)(x)

	model_weights =  model.get_weights()
	model = Model(inputs = transfer.inputs, outputs = outputs)
	model.set_weights(model_weights)
	return model