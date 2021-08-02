import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Model

from tensorflow.keras.layers import MaxPool2D, Input, Dense, GlobalAveragePooling2D, Conv2D, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.activations import relu

#Import Cam functions
from tf_keras_vis.gradcam import GradcamPlusPlus, Gradcam
from tf_keras_vis.scorecam import Scorecam

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import cv2

#Define numpy error handler
def numpy_error_handler(type, flag):
	print("\n[WARN] Floating point error (%s)" % (type))

np.seterrcall(numpy_error_handler)

#Define Constants
IMAGE_SIZE = (256, 256)
TOTAL_CLASSES = 199
BATCH_SIZE = 4

#CAMERAS Function
def cameras_cam(model, image_array, penultimate_layer=-1, input_resolutions=None, label_index=None, change_input_shape=None, activation_modifier=None):
	"""Args:
		 model: A tensorflow.keras.Model object, 
				The model
		 image_array: An array
				The input image
		 penultimate_layer: An interger
				The index of last convolutional layer
		 input_resolutions: A tuple object (start, end, step) 
				Range of image resolutions for cameras, 
				eg. if input_resolutions = (256, 1000, 100), 
				then resolutions are (256X256), (356X356), (456X456).......(1000X1000)
		 label_index: An integer
				The class index for which we are interested to obtain cam.
		 change_input_shape: A tuple object, 
				The function to change input shape of model,
		 activation_modifier: A function 
		 		The function which modifies Class Activation Map (CAM). Defaults to
                lambda cam: K.relu(cam).

	   Return: An array,
			The heatmap produced from cameras"""

	#If input resolutions not provided
	np.seterr(all='call')
	if input_resolutions is None:
		input_resolutions = list(range(256, 1000, 100))

	#If label index not provided
	if label_index is None:
		preds = model.predict(np.expand_dims(image_array, axis=0))
		label_index = np.argmax(preds)

	#Obtain gradients and activations for each input resolution
	features = {}
	gradients = {}
	for i, input_resolution in enumerate(input_resolutions):
		#Change the model input to current input_resolution
		new_model = change_input_shape(model=model, new_input_shape=(input_resolution, input_resolution, 3))

		#Resize image to current input_resolution
		resized_image_array = cv2.resize(image_array, (input_resolution, input_resolution), interpolation=cv2.INTER_LINEAR)
		resized_image_array = np.expand_dims(resized_image_array, axis=0)
		
		#Create Grad model
		grad_model = tf.keras.models.Model([new_model.inputs], [new_model.layers[penultimate_layer].output, new_model.output])
		with tf.GradientTape() as tape:
			layer_activation, preds = grad_model(resized_image_array)
			class_channel = preds[:, label_index]

		#Obtain gradients and activations
		grads = tape.gradient(class_channel, layer_activation)
		activations = layer_activation[0]

		features[input_resolution] = activations
		gradients[input_resolution] = grads
		
	#Obtain saliency/heatmap
	ResizeLayer = Resizing(height=input_resolutions[0],  width=input_resolutions[0], interpolation="bilinear")
	up_sampled_features = None
	up_sampled_gradients = None
	count = 0
	for resolution in features.keys():
		activations = features[resolution]
		grads = gradients[resolution]

		if up_sampled_features is None or up_sampled_gradients is None:
			up_sampled_features = ResizeLayer(activations).numpy()
			up_sampled_gradients = ResizeLayer(grads).numpy()
		else:
			up_sampled_features += ResizeLayer(activations).numpy()
			up_sampled_gradients += ResizeLayer(grads).numpy()

		count += 1

	fmaps = up_sampled_features / count
	grads = up_sampled_gradients / count

	fmaps = fmaps.squeeze()
	grads = grads.squeeze()

	saliency = relu((fmaps * grads).sum(axis=2)).numpy()
	saliency = saliency - saliency.min()
	cam = saliency / saliency.max()

	if not activation_modifier is  None:
		cam = activation_modifier(cam)

	return cam

#Generate Cam Wrapper Function to all Cams
def gen_cam(cam_type, model, image_array, activation_layer_index, label_index=None, activation_modifier=None, change_input_shape=None, input_resolutions=None):
	"""Args:
		cam_type: A str
			Available options are 'gradcam', 'gradcampp', 'scorecam', cameras.
		model: A tensorflow.keras.Model object, 
			The model
		image_array: An array
			The input image
		activation_layer_index: An interger
			The index of last convolutional layer
		label_index: An integer
			The class index for which we are interested to obtain cam.
		activation_modifier: A function 
		 	The function which modifies Class Activation Map (CAM). Defaults to
            lambda cam: K.relu(cam).

        *For CAMERAS ONLY*
        change_input_shape: A tuple object, 
			The function to change input shape of model
        input_resolutions: A tuple object (start, end, step) 
			Range of image resolutions for cameras, 
			eg. if input_resolutions = (256, 1000, 100), 
			then resolutions are (256X256), (356X356), (456X456).......(1000X1000)

	   Return: An array,
			The heatmap produced from cameras"""

	#Gen Cam Heat Maps
	if cam_type == "gradcam":
		gradcam = Gradcam(model, model_modifier=None, clone=False)
		heatmap = gradcam(score_function(label_index), 
											image_array, 
											penultimate_layer=activation_layer_index, 
											seek_penultimate_conv_layer=False, 
											activation_modifier=None)[0]
	elif cam_type == "gradcampp":
		gradcampp = GradcamPlusPlus(model, model_modifier=None, clone=False)
		heatmap = gradcampp(score_function(label_index), 
											image_array, 
											penultimate_layer=activation_layer_index, 
											seek_penultimate_conv_layer=False, 
											activation_modifier=None)[0]
	elif cam_type == "scorecam":
		scorecam = Scorecam(model, model_modifier=None, clone=False)
		heatmap = scorecam(score_function(label_index),
											image_array, 
											penultimate_layer=activation_layer_index, 
											seek_penultimate_conv_layer=False, 
											activation_modifier=None,
											max_N=10)[0]
	elif cam_type == 'cameras':
		if change_input_shape == None:
			print("[ERROR] Call to get_cam(cam_type='cameras',...) requires argument 'change_input_shape'")
			exit()
		heatmap = cameras_cam(model=model, 
									image_array=image_array,
									penultimate_layer=activation_layer_index,
									change_input_shape=change_input_shape,
									activation_modifier=None,
									label_index=label_index,
									input_resolutions=input_resolutions)

	return heatmap

#Change input shape function for all standard models 
def get_change_input_func(model_api_func):
	def change_input_shape1(model, new_input_shape):
		#Define new model with changed input shape
		transfer = model_api_func(include_top = False, weights=None, input_tensor=Input(shape=(new_input_shape[0], new_input_shape[1], 3)))
		x = GlobalAveragePooling2D()(transfer.layers[-1].output)
		x = Dropout(0.5)(x)
		outputs = Dense(TOTAL_CLASSES, activation='softmax')(x)
		#Load weights into new model
		model_weights =  model.get_weights()
		new_model = Model(inputs = transfer.inputs, outputs = outputs)
		new_model.set_weights(model_weights)   
		return new_model

	return change_input_shape1

#Change input shape function for our Custom-CNN
def change_input_shape2(model, new_input_shape):
	inputs = Input(shape=(new_input_shape[0], new_input_shape[1], 3))
	
	x = Conv2D(32, 7, activation="relu", kernel_initializer='random_uniform')(inputs)
	x = MaxPool2D((2,2))(x)

	x = Conv2D(64, 5, activation="relu", kernel_initializer='random_uniform')(x)
	x = MaxPool2D((2,2))(x)

	x = Conv2D(64, 5, activation="relu", kernel_initializer='random_uniform')(x)
	x = MaxPool2D((2,2))(x)

	x = Conv2D(64, 5, activation="relu", kernel_initializer='random_uniform')(x)
	x = MaxPool2D((2,2))(x)

	x = Conv2D(512, 3, activation="relu", kernel_initializer='random_uniform')(x)
	x = MaxPool2D((2,2))(x)

	x = Conv2D(512, 3, activation="relu", kernel_initializer='random_uniform')(x)
		
	x = GlobalAveragePooling2D()(x)
	outputs = Dense(199)(x)

	model_weights = model.get_weights()
	new_model = Model(inputs=inputs, outputs=outputs)
	new_model.set_weights(model_weights)
	return new_model

#Convert PIL image to numpy array
def get_img_array(img, target_size=None, preprocess_input = None, rescale = None):
	#Convert to array
	array = img_to_array(img)

	#Resize
	if target_size != None:
		array = cv2.resize(array, target_size)

	#Apply preprocessing
	if preprocess_input != None: 
		array = preprocess_input(array)

	if rescale != None:
		array = array*rescale

	return array

#Superimpose heatmap on original image
def get_superimposed_image(img, heatmap, alpha=0.5):
	# Rescale heatmap to a range 0-255
	heatmap = np.uint8(255 * heatmap)

	# Use jet colormap to colorize heatmap
	jet = cm.get_cmap("jet")

	# Use RGB values of the colormap
	jet_colors = jet(np.arange(256))[:, :3]
	jet_heatmap = jet_colors[heatmap]

	# Create an image with RGB colorized heatmap
	jet_heatmap = array_to_img(jet_heatmap)
	jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
	jet_heatmap = img_to_array(jet_heatmap)

	# Superimpose the heatmap on original image
	superimposed_img = jet_heatmap * alpha + img
	superimposed_img = array_to_img(superimposed_img)

	return superimposed_img

#Get Score Function
def score_function(pred_index):
	return lambda output: (output[0][pred_index])