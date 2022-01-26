import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.activations import relu

#Import Cam functions
from tf_keras_vis.gradcam import GradcamPlusPlus, Gradcam
from tf_keras_vis.scorecam import Scorecam

import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import cv2

#Define numpy error handler
def numpy_error_handler(type, flag):
	print("\n[WARN] Floating point error (%s)" % (type))

np.seterrcall(numpy_error_handler)

#CAMERAS Function
def cameras_cam(model, image_array, penultimate_layer=-1, input_resolutions=None, label_index=None, activation_modifier=None):
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
				The class index for which we are interested to obtain cam
		 activation_modifier: A function 
		 		The function which modifies Class Activation Map (CAM). Defaults to
                lambda cam: K.relu(cam).

	   Return: An array,
			The heatmap produced from cameras"""

	#If input resolutions not provided
	np.seterr(all='call')
	if input_resolutions is None:
		input_resolutions = list(range(model.input.shape[1], model.input.shape[1] + 701, 100))

	#If label index not provided
	if label_index is None:
		preds = model.predict(np.expand_dims(image_array, axis=0))
		label_index = np.argmax(preds[0])

	#Obtain gradients and activations for each input resolution
	features = {}
	gradients = {}
	for i, input_resolution in enumerate(input_resolutions):
		#Change the model input to current input_resolution
		input_tensors = Input(shape=(input_resolution, input_resolution, 3))
		new_model = tf.keras.models.clone_model(model, input_tensors=input_tensors, clone_function=None)
		new_model.set_weights(model.get_weights())

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

	return cam

#Generate Cam Wrapper Function for all Cams
def gen_cam(cam_type, model, image_array, activation_layer_index, label_index=None):
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
	elif cam_type == "f-scorecam":
		scorecam = Scorecam(model, model_modifier=None, clone=False)
		heatmap = scorecam(score_function(label_index),
											image_array, 
											penultimate_layer=activation_layer_index, 
											seek_penultimate_conv_layer=False, 
											activation_modifier=None,
											max_N=10)[0]
	elif cam_type == "scorecam":
		scorecam = Scorecam(model, model_modifier=None, clone=False)
		heatmap = scorecam(score_function(label_index),
											image_array, 
											penultimate_layer=activation_layer_index, 
											seek_penultimate_conv_layer=False, 
											activation_modifier=None)[0]
	elif cam_type == 'cameras':
		heatmap = cameras_cam(model=model, 
									image_array=image_array,
									penultimate_layer=activation_layer_index,
									label_index=label_index
									)
	elif cam_type == 'guidedbp':
		heatmap = guidedBP(model=model, 
									image_array=image_array,
									penultimate_layer=activation_layer_index,
									label_index=label_index
									)

	return heatmap

#Convert PIL image to numpy array
def get_img_array(img, target_size=None, preprocess_input = None):
	#Convert to array
	array = img_to_array(img)

	#Resize
	if target_size != None:
		array = cv2.resize(array, target_size)

	#Apply preprocessing
	if preprocess_input != None: 
		array = preprocess_input(array)

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
	jet_heatmap = jet_heatmap.resize((img.shape[0], img.shape[1]))
	jet_heatmap = img_to_array(jet_heatmap)

	# Superimpose the heatmap on original image
	superimposed_img = jet_heatmap * alpha + img
	superimposed_img = array_to_img(superimposed_img)

	return superimposed_img

#Get Score Function
def score_function(pred_index):
	return lambda output: (output[0][pred_index])


def guidedBP(model, image_array, penultimate_layer=-1, label_index=None, activation_modifier=None):
	"""Args:
		 model: A tensorflow.keras.Model object, 
				The model
		 image_array: An array
				The input image
		 penultimate_layer: An interger
				The index of last convolutional layer
		label_index: An integer
				The class index for which we are interested to obtain cam
		 activation_modifier: A function 
		 		The function which modifies Class Activation Map (CAM). Defaults to
                lambda cam: K.relu(cam).

	   Return: An array,
			The heatmap produced from cameras"""

	image_array = np.expand_dims(image_array, axis=0)

	#Create Grad model
	grad_model = tf.keras.models.Model([model.inputs], [model.layers[penultimate_layer].output, model.output])

	#Get all activation layers
	layer_dict = [layer for layer in grad_model.layers[1:] if hasattr(layer,'activation')]

	#Define Guidede Relu activation function
	@tf.custom_gradient
	def guidedRelu(x):
		def grad(dy):
			return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
		return tf.nn.relu(x), grad

	#Apply GuidedRelu to all activation layers
	for layer in layer_dict:
		if layer.activation == tf.keras.activations.relu:
			layer.activation = guidedRelu
	
	with tf.GradientTape() as tape:
		inputs = tf.cast(image_array, tf.float32)
		tape.watch(inputs)
		outputs = grad_model(inputs)[0]
		
	guided_back_prop = tape.gradient(outputs,inputs)[0]
	gb_viz = np.dstack((
				guided_back_prop[:, :, 0],
				guided_back_prop[:, :, 1],
				guided_back_prop[:, :, 2]
	))       
	gb_viz -= np.min(gb_viz)
	gb_viz /= gb_viz.max()

	return gb_viz