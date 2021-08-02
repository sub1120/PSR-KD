from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Average, Dropout, Layer
from tensorflow.keras.models import Model, model_from_json

#Import models API
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_resnet_v2 
from tensorflow.keras.applications import resnet_v2 
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import mobilenet_v2

import os

#Define Constants
IMAGE_SIZE = (256, 256)
TOTAL_CLASSES = 199

#Define Teacher path and ensemble
TEACHER_MODEL_PATH = 'models/Teacher_Models/'
TO_ENSEMBLE = ['EfficientNetB7', 'DenseNet201']

#Get Ensemble Function with given input shape
def get_ensemble(models, input_shape=IMAGE_SIZE):
	emodels = []
	ensemble_input = Input(shape=(input_shape[0], input_shape[1], 3), name='ensemble_input')
	for j in range(len(TO_ENSEMBLE)):
		emodel = models[TO_ENSEMBLE[j]]['model']
		emodel = add_preprocess_layer(emodel, models[TO_ENSEMBLE[j]]['api_func'], models[TO_ENSEMBLE[j]]['preprocess_layer'], ensemble_input, input_shape)
		emodel = rename_all_layers(emodel, TO_ENSEMBLE[j])
		emodels.append(emodel)

	outputs = [m.output for m in emodels]
	y = Average()(outputs) 
	model = Model(inputs=ensemble_input , outputs=y)
	return model

#Add preprocessing layer to model
def add_preprocess_layer(model, api_func, pre_layer, ensemble_input, input_shape):
	transfer = api_func(include_top = False, weights = None, input_tensor=Input(shape=(input_shape[0], input_shape[1], 3)))
	x = pre_layer()(ensemble_input)
	x = transfer(x)
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.5)(x)
	outputs = Dense(TOTAL_CLASSES)(x)

	model_weights = model.get_weights()
	model = Model(inputs = ensemble_input, outputs = outputs)
	model.set_weights(model_weights)
	return model

#Rename all layers except input layer of model
def rename_all_layers(model, file):
	for i in range(1, len(model.layers)):
		model.layers[i]._name = model.layers[i]._name + '_' + file

	return model

#Define Preprocessing Layers
class ENB7_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(ENB7_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = efficientnet.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(ENB7_PreprocessLayer, self).get_config()
		return config

class D201_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(D201_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = densenet.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(D201_PreprocessLayer, self).get_config()
		return config

class ENB7_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(ENB7_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = efficientnet.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(ENB7_PreprocessLayer, self).get_config()
		return config

class D201_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(D201_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = densenet.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(D201_PreprocessLayer, self).get_config()
		return config

class NNL_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(NNL_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = nasnet.preprocess_input

	def call(self, input):
		return nasnet.preprocess_input(input)

	def get_config(self):
		config = super(NNL_PreprocessLayer, self).get_config()
		return config

class RNV2_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(RNV2_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = resnet_v2.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(RNV2_PreprocessLayer, self).get_config()
		return config

class IRNV2_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(IRNV2_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = inception_resnet_v2.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(IRNV2_PreprocessLayer, self).get_config()
		return config

class XCEP_PreprocessLayer(Layer):
	def __init__(self, name="preprocess", **kwargs):
		super(XCEP_PreprocessLayer, self).__init__(name=name, **kwargs)
		self.preprocess = xception.preprocess_input

	def call(self, input):
		return self.preprocess(input)

	def get_config(self):
		config = super(XCEP_PreprocessLayer, self).get_config()
		return config