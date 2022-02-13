import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf

#Import custom modules
from utils.cam_utils import get_img_array, get_superimposed_image, gen_cam
from utils.load_utils import init_models, create_data_generator
from utils.eval_utils import evaluate, cost_compute

#Import other required modules
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

#Define Constants
BATCH_SIZE = 4
NUM_CLASSES = 199
FONT_SIZE = 10
DPI = 300

#Set model paths
MODEL_PATH = 'models/'

#SET DATA PATHS
MAIN_DATA_DIR = "ds/"
TRAIN_DATA_DIR = MAIN_DATA_DIR + "train/"
TEST_DATA_DIR = MAIN_DATA_DIR + "test/"
VALIDATION_DATA_DIR = MAIN_DATA_DIR + "val/"

#Input Modes
MODES = ['IMAGE MODE', 'VIDEO MODE']

#Save Figure Function
def savefigure(directory, fig_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + fig_name + '.tiff', bbox_inches='tight', dpi=DPI, format='tiff')

#Create {index:classs_name} dictionary
def index_class_dict():
	file_names = os.listdir('ds/test/')
	file_names = sorted(file_names)
	index_class = {}
	for i in range(len(file_names)):
		index_class[i] = file_names[i]

	return index_class

#Softmax function
def softmax(arr):
	for i in range(len(arr)):
		ex =np.exp(arr[i])
		arr[i] = ex/np.sum(ex)
	return arr

def produce_cam(model_name, is_gradcam, is_gradcamplus, is_f_scorecam, is_scorecam, is_camerascam, is_guidedbp, input_files, enable_gpu):
	
	#Check gpu availability
	if enable_gpu:
		gpus = len(tf.config.list_physical_devices('GPU'))
		print('[INFO] Tensorflow recognized {} GPUs'.format(gpus))
	else:
		tf.config.set_visible_devices([], 'GPU')

	cam_count = [is_scorecam, is_f_scorecam, is_gradcamplus, is_gradcam, is_camerascam, is_guidedbp].count(True)

	#Load {index:classs_name} dictionary
	index_class = index_class_dict()

	#Load required model
	all_models = init_models(MODEL_PATH, model_name)
	model = all_models[model_name]['model']
	model.layers[-1].activation = None

	#Set preprocessing
	preprocess_func = all_models[model_name]['preprocess_func']
	image_size = all_models[model_name]['image_size']
	activation_layer_index = -4

	#Load and preprocess Images
	input_images = []
	for j in tqdm(range(len(input_files)),desc = "[INFO] Loading Images", ncols=80):
		file_name = input_files[j]
		image = load_img(file_name)
		image_array = get_img_array(image, image_size, preprocess_func)
		input_images.append(image_array)

	#Get predictions
	predictions = [] 
	scores = []
	for j in tqdm(range(len(input_files)),desc = "[INFO] Getting Predictions", ncols=80):
		image_array = input_images[j]
		image_array = np.expand_dims(image_array, axis=0)
		preds = model.predict(image_array)
		preds = softmax(preds)
		pred_index = np.argmax(preds[0])
		score = "{0:.3f}%".format(preds[0][pred_index]*100)
		predictions.append(pred_index)
		scores.append(score)

	#Generate Cams
	cam_images = {}
	for j in tqdm(range(len(input_files)),desc = "[INFO] Generating Cams", ncols=80):
		file_name = input_files[j]
		image_array = input_images[j]
		pred_index = predictions[j]
		cam_images[file_name] = {}

		#Grad-Cam
		if is_gradcam:
			cam_images[file_name]['gradcam'] = gen_cam(cam_type='gradcam', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
		#Grad-Cam++
		if is_gradcamplus:
			cam_images[file_name]['gradcampp'] = gen_cam(cam_type='gradcampp', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)	
		#FasterScore-Cam++
		if is_f_scorecam:
			cam_images[file_name]['f-scorecam'] = gen_cam(cam_type='f-scorecam', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
		#Scorecam
		if is_scorecam:
			cam_images[file_name]['scorecam'] = gen_cam(cam_type='scorecam', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
		#CAMERAS
		if is_camerascam:
			cam_images[file_name]['cameras'] = gen_cam(cam_type='cameras', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)

		if is_guidedbp:
			cam_images[file_name]['guidedbp'] = gen_cam(cam_type='guidedbp', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
		
	#Plot Cams
	print("[INFO] Plotting Cams")
	OUTPUT_PATH = 'out/Cams/' 

	i=0
	rows, cols = (len(cam_images), cam_count + 1)
	plt.figure(figsize=(12, 12))
	for k in range(len(input_files)):
		file_name = input_files[k]
		actual_class = file_name.split('/')[-1].split('.')[0]

		org_img = load_img(file_name)
		org_img = get_img_array(org_img, image_size)
		pred_index = predictions[k]
		score = scores[k]
					
		j = 1
		#Plot Original Image
		plt.subplot(rows, cols, i + j)
		plt.title("Prediction: " + index_class[pred_index] +' ('+score+')', fontsize=FONT_SIZE)
		plt.imshow(org_img/255.0)
		plt.xticks([])
		plt.yticks([])
		j += 1
				
		#Plot Superimposed heatmap on Original Image
		if is_gradcam:
			gradcam_super_img = get_superimposed_image(org_img, cam_images[file_name]['gradcam'])
			plt.subplot(rows, cols, i + j)
			plt.imshow(gradcam_super_img)
			plt.xticks([])
			plt.yticks([])
			plt.title("Grad-CAM", fontsize=FONT_SIZE)
			j += 1
				
		if is_gradcamplus:
			gradcampp_super_img = get_superimposed_image(org_img, cam_images[file_name]['gradcampp'])
			plt.subplot(rows, cols, i + j)
			plt.imshow(gradcampp_super_img)
			plt.xticks([])
			plt.yticks([])
			plt.title("Grad-Cam++", fontsize=FONT_SIZE)
			j += 1
		
		if is_f_scorecam:
			f_scorecam_super_img = get_superimposed_image(org_img, cam_images[file_name]['f-scorecam'])
			plt.subplot(rows, cols, i + j)
			plt.imshow(f_scorecam_super_img)
			plt.xticks([])
			plt.yticks([])
			plt.title("Faster Score-Cam", fontsize=FONT_SIZE)
			j += 1

		if is_scorecam:
			scorecam_super_img = get_superimposed_image(org_img, cam_images[file_name]['scorecam'])
			plt.subplot(rows, cols, i + j)
			plt.imshow(scorecam_super_img)
			plt.xticks([])
			plt.yticks([])
			plt.title("Score-Cam", fontsize=FONT_SIZE)
			j += 1
					
		if is_camerascam:
			cameras_super_img = get_superimposed_image(org_img, cam_images[file_name]['cameras'])
			plt.subplot(rows, cols, i + j)
			plt.imshow(cameras_super_img)
			plt.xticks([])
			plt.yticks([])
			plt.title("Cameras-Cam", fontsize=FONT_SIZE)
			j += 1

		if is_guidedbp:
			plt.subplot(rows, cols, i + j)
			plt.imshow(cam_images[file_name]['guidedbp'])
			plt.xticks([])
			plt.yticks([])
			plt.title("Guided-Bp", fontsize=FONT_SIZE)
			j += 1
			
		i += (cam_count + 1)

	savefigure(OUTPUT_PATH, model_name + '-' + actual_class)
	plt.show()		
	plt.close('all')
					
	print("[INFO] Done.")


def evaluate_model(model_name, enable_gpu):
	#Check gpu availability
	if enable_gpu:
		gpus = len(tf.config.list_physical_devices('GPU'))
		print('[INFO] Tensorflow recognized {} GPUs'.format(gpus))
	else:
		tf.config.set_visible_devices([], 'GPU')

	#Load required model
	all_models = init_models(MODEL_PATH, model_name)
	model = all_models[model_name]['model']

	#compute cost
	print("\n[INFO] Model Cost")
	cost_compute(MODEL_PATH + model_name + '/' + model_name + '.h5')

	#Set preprocessing
	preprocess_func = all_models[model_name]['preprocess_func']
	image_size = all_models[model_name]['image_size']

	#Test on Validation data
	print("\n[INFO] Model Validation ")
	validation_generator = create_data_generator(data_path=VALIDATION_DATA_DIR, 
													input_shape=image_size, 
													batch_size=BATCH_SIZE, 
													pre_process=preprocess_func)
	evaluate(model, validation_generator)
	
	#Test on Test data
	print("\n[INFO] Model Testing ")
	test_generator = create_data_generator(data_path=TEST_DATA_DIR, 
												input_shape=image_size, 
												batch_size=BATCH_SIZE, 
												pre_process=preprocess_func)
	evaluate(model, test_generator)		
