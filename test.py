import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf

#Import custom modules
from utils.cam_utils import get_img_array, get_superimposed_image, gen_cam,  get_change_input_func, change_input_shape2
from utils.load_utils import init_models

#Import other required modules
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
import numpy as np
import cv2

#Define Constants
IMAGE_SIZE = (256, 256)
TOTAL_CLASSES = 199
BATCH_SIZE = 4
FONT_SIZE = 10
DPI = 600

#Input Modes
MODES = ['IMAGE MODE', 'VIDEO MODE']

#List all model files
DISTILLED_STUDENT_FILES = ['KD-DenseNet121', 'KD-EfficientNetB0', 'KD-NASNetMobile', 'KD-MobileNetV2', 'KD-Custom-CNN']
NORMAL_STUDENT_FILES = ['DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'Custom-CNN']
TEACHER_FILES = ['DenseNet201', 'Xception', 'InceptionResNetV2', 'ResNet152V2', 'EfficientNetB7', 'NASNetLarge', 'EnsembleModel']
MODEL_FILES = DISTILLED_STUDENT_FILES + NORMAL_STUDENT_FILES +  TEACHER_FILES

#Save Figure Function
def savefigure(directory, fig_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + fig_name + '.png', bbox_inches='tight', dpi=DPI, format='png')

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

def process_frame(org_img_array, preprocess_func, rescale):
	img_array = cv2.cvtColor(org_img_array, cv2.COLOR_BGR2GRAY)							#Convert image to Gray
	_,img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)					#Apply thresholding
	img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)								#Convert back image to RGB
	img_array = get_img_array(img_array, IMAGE_SIZE, preprocess_func, rescale)			#Apply preprocesing of model
	return img_array

def main(is_video_mode, model_name, is_gradcam, is_gradcamplus, is_scorecam, is_camerascam, input_files):
	tf.config.set_visible_devices([], 'GPU')
	cam_count = [is_scorecam, is_gradcamplus, is_gradcam, is_camerascam].count(True)

	#Load {index:classs_name} dictionary
	index_class = index_class_dict()

	#Load required model
	all_models = init_models([model_name])
	model = all_models[model_name]['model']
	
	#Set preprocessing
	preprocess_func = all_models[model_name]['preprocess_func']
	rescale = all_models[model_name]['rescale']

	#Set last_conv_layer_index and change_input_shape function
	if model_name in ['Custom-CNN', 'KD-Custom-CNN']:
		activation_layer_index = -3
		change_input_shape = change_input_shape2
	else:
		activation_layer_index = -4
		change_input_shape = get_change_input_func(all_models[model_name]['api_func'])

	if is_video_mode:
		#Get Activation heatmap
		if is_gradcamplus:
			print("[INFO] Set CAM: Grad-Cam++")
		elif is_gradcam:
			print("[INFO] Set CAM: Grad-Cam")
		elif is_scorecam:
			print("[INFO] Set CAM: Faster Score-Cam")
		elif is_camerascam:
			print("[INFO] Set CAM: CAMERAS-Cam")
		
		#Start Capturing
		print("[INFO] Starting Camera")
		capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		if not capture.isOpened():
			print("[ERROR]: Could not open video device")

		print("[INFO] Press 'q' to stop Camera")
		start = (175,125)
		end = (475,400)
		loop = 0
		while(True):     
			ret, frame = capture.read()
			font = cv2.FONT_HERSHEY_SIMPLEX

			#Take yellow rectangle only
			org_image_array = frame[start[1]:end[1], start[0]:end[0]]								
			
			#Preprocess frame
			image_array = process_frame(org_image_array, preprocess_func, rescale)
										
			#Prediction
			preds = model.predict(np.expand_dims(image_array, axis=0))
			soft_preds = softmax(preds)
			pred_index = np.argmax(soft_preds[0])
			score = "{0:.3f}%".format(soft_preds[0][pred_index]*100)
			
			#Get heatmap
			if loop%10==0:
				if is_gradcamplus:
					heatmap =  gen_cam(cam_type='gradcam', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
				elif is_gradcam:
					heatmap =  gen_cam(cam_type='gradcampp', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
				elif is_scorecam:
					heatmap =  gen_cam(cam_type='scorecam', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
				elif is_camerascam:
					heatmap = gen_cam(cam_type='cameras', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index, change_input_shape=change_input_shape)

			if cam_count == 1:
				#Create superimposed heatmap image frame
				super_image_array = get_superimposed_image(org_image_array, heatmap)
				super_image_array = np.array(super_image_array)
				top , bottom, left, right  = start[1], frame.shape[0] - end[1], start[0], frame.shape[1] - end[0]
				super_image_array = cv2.copyMakeBorder(super_image_array, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
				org_image_array = cv2.copyMakeBorder(org_image_array, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
				super_frame = frame + super_image_array - org_image_array
			else:
				super_frame = frame

			cv2.putText(super_frame, 'Prediction: ' + index_class[pred_index], (20, 30), font, 1, (0, 255, 255), 2, lineType = cv2.LINE_4)
			cv2.putText(super_frame, 'Confidence: ' + score, (20, 65), font, 1, (0, 255, 255), 2, lineType = cv2.LINE_4)
			cv2.rectangle(super_frame, start, end, (0, 255, 255), 2)
			cv2.imshow('PSR', super_frame)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			loop = (loop + 1)%100

		capture.release()
		cv2.destroyAllWindows()
	else:
		#Load and preprocess Images
		input_images = []
		for j in tqdm(range(len(input_files)),desc = "[INFO] Loading Images", ncols=80):
			file_name = input_files[j]
			image = load_img(file_name)
			image_array = get_img_array(image, IMAGE_SIZE, preprocess_func, rescale)
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
			if is_scorecam:
				cam_images[file_name]['scorecam'] = gen_cam(cam_type='scorecam', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index)
			#CAMERAS
			if is_camerascam:
				cam_images[file_name]['cameras'] = gen_cam(cam_type='cameras', model=model, image_array=image_array, label_index=pred_index, activation_layer_index=activation_layer_index, change_input_shape=change_input_shape)

		#Plot Cams
		i = 0
		rows, cols = (len(cam_images), cam_count + 1)
		plt.figure(figsize=(12, 12))
		plt.title("CAMS")
		for j in tqdm(range(len(input_files)),desc = "[INFO] Plotting Cams", ncols=80):
			file_name = input_files[j]
			org_img = load_img(file_name)
			org_img = get_img_array(org_img, IMAGE_SIZE)
			pred_index = predictions[j]
			score = scores[j]
			
			j = 1
			#Plot Original Image
			plt.subplot(rows, cols, i + j)
			plt.imshow(org_img/255.0)
			plt.xticks([])
			plt.yticks([])
			plt.title("Predicted: " + index_class[pred_index] +' ('+score+')', fontsize=FONT_SIZE)
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
			
			if is_scorecam:
				scorecam_super_img = get_superimposed_image(org_img, cam_images[file_name]['scorecam'])
				plt.subplot(rows, cols, i + j)
				plt.imshow(scorecam_super_img)
				plt.xticks([])
				plt.yticks([])
				plt.title("Faster Score-Cam", fontsize=FONT_SIZE)
				j += 1
			
			if is_camerascam:
				cameras_super_img = get_superimposed_image(org_img, cam_images[file_name]['cameras'])
				plt.subplot(rows, cols, i + j)
				plt.imshow(cameras_super_img)
				plt.xticks([])
				plt.yticks([])
				plt.title("Cameras-Cam", fontsize=FONT_SIZE)
				j += 1

			i += (cam_count + 1)
		output_path = 'out/Cams'
		savefigure(output_path, model_name)
		plt.show()	
		plt.close('all')
	print("[INFO] Done.")