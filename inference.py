import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import accuracy_score
from utils.load_utils import init_models
import tensorflow as tf
import argparse
import numpy as np
import math
import time

#List all model files
DISTILLED_STUDENT_FILES = ['KD-DenseNet121', 'KD-EfficientNetB0', 'KD-NASNetMobile', 'KD-MobileNetV2', 'KD-Custom-CNN']
NORMAL_STUDENT_FILES = ['DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'Custom-CNN']
TEACHER_FILES = ['DenseNet201', 'Xception', 'InceptionResNetV2', 'ResNet152V2', 'EfficientNetB7', 'NASNetLarge', 'EnsembleModel']
MODEL_FILES = DISTILLED_STUDENT_FILES + NORMAL_STUDENT_FILES +  TEACHER_FILES

VAL_DATA_PATH = 'ds/val'
TEST_DATA_PATH = 'ds/test'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

def init_arg_parser():
	arg_parser = argparse.ArgumentParser(description='Calculate Inference Time', epilog='')
	arg_parser.add_argument("model_index", help="Give model index")
	arg_parser.add_argument("-g", "--enable_gpu", help="Enable GPU", action="store_true", default=False)
	return arg_parser

def load_data(datapath):
	#Load validation dataset
	data = []
	for current_dir in os.listdir(datapath):
		for current_file in os.listdir(datapath + '/' + current_dir):
			image_path = os.path.join(datapath + '/' + current_dir, current_file)
			data.append(load_img(image_path, IMAGE_SIZE))

	return data

#Get data from generator function
def get_data(generator, nb_samples):  
	from tqdm import tqdm
	x = []
	y = []

	with tqdm(total=nb_samples, ncols=60) as pbar:
		for i in range(math.ceil(nb_samples/BATCH_SIZE)):
			new_x, new_y = generator[i][0], generator[i][1]
			x.extend(new_x)
			y.extend(new_y)
			pbar.update(len(new_y))

	x = np.array(x)
	y = np.array(y)
	return x, y

#Softmax Function
def softmax(arr):
    for i in range(len(arr)):
        ex =np.exp(arr[i])
        arr[i] = ex/np.sum(ex)
    return arr
def cal_inf(data):
	#Calculate Val inference time
	start_time = time.time()
	y_prob = model.predict(data)
	end_time = time.time()
	diff = end_time - start_time 
	return diff, y_prob

if __name__ == '__main__':

	#Initialize argument parser
	arg_parser = init_arg_parser()
	args = arg_parser.parse_args()
	model_index = args.model_index
	enable_gpu = args.enable_gpu

	#Check gpu availability
	if enable_gpu:
		gpus = len(tf.config.list_physical_devices('GPU'))
		print('[INFO] Tensorflow recognized {} GPUs'.format(gpus))
	else:
		tf.config.set_visible_devices([], 'GPU')

	#Create {index:model_name} dictionary
	index_model = {}
	for i in range(len(MODEL_FILES)):
		index_model[i] = MODEL_FILES[i]

	#Obtain model name from model index
	model_name = index_model[int(model_index)]

	#Load required model
	all_models = init_models([model_name])
	model = all_models[model_name]['model']
	preprocess_input = all_models[model_name]['preprocess_func']

	#Create Generator from datapath
	print('\n[INFO] NUMBER OF VALIDATION & TEST SAMPLES')
	datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
	validation_generator = datagen.flow_from_directory(VAL_DATA_PATH, 
															target_size=IMAGE_SIZE, 
															batch_size=BATCH_SIZE, 
															class_mode='sparse',
															shuffle=False)
	test_generator = datagen.flow_from_directory(TEST_DATA_PATH, 
													  target_size=IMAGE_SIZE, 
													  batch_size=BATCH_SIZE, 
													  class_mode='sparse',
													  shuffle=False)

	nb_validation_samples = len(validation_generator.filenames)
	nb_test_samples = len(test_generator.filenames)
	num_classes= len(validation_generator.class_indices)

	print("\n[INFO] Loading Validation Data")
	x_val, y_val = get_data(validation_generator, nb_validation_samples)

	print("[INFO] Loading Testing Data")
	x_test, y_test = get_data(test_generator, nb_test_samples)

	print("\n[INFO] Calculating Inference Time.....")
	diff, y_prob = cal_inf(x_val)
	diff = '{0:.4f}'.format(diff)

	#Calculate Val Accuracy
	y_soft_prob = softmax(y_prob)
	y_pred = y_soft_prob.argmax(axis=-1)
	acc = '{0:.4f}%'.format(accuracy_score(y_val, y_pred)*100)

	print("Validation Time: ", diff, " sec")
	print("Validation Accuracy: ", acc)

	#Calculate Test inference time
	diff, y_prob = cal_inf(x_test)
	diff = '{0:.4f}'.format(diff)

	#Calculate Test Accuracy
	y_soft_prob = softmax(y_prob)
	y_pred = y_soft_prob.argmax(axis=-1)
	acc = '{0:.4f}%'.format(accuracy_score(y_test, y_pred)*100)

	print("\nTest Time: ", diff, " sec")
	print("Test Accuracy: ", acc)