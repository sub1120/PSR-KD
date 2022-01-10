#Import models API
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_resnet_v2 
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import resnet_v2 
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import mobilenet_v2

class EN_PreprocessLayer(Layer):
    def __init__(self, name="preprocess", **kwargs):
        super(EN_PreprocessLayer, self).__init__(name=name, **kwargs)
        self.preprocess = efficientnet.preprocess_input

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(EN_PreprocessLayer, self).get_config()
        return config

class Dense_PreprocessLayer(Layer):
    def __init__(self, name="preprocess", **kwargs):
        super(Dense_PreprocessLayer, self).__init__(name=name, **kwargs)
        self.preprocess = densenet.preprocess_input

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(Dense_PreprocessLayer, self).get_config()
        return config

class INV3_PreprocessLayer(Layer):
    def __init__(self, name="preprocess", **kwargs):
        super(INV3_PreprocessLayer, self).__init__(name=name, **kwargs)
        self.preprocess = inception_v3.preprocess_input

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(INV3_PreprocessLayer, self).get_config()
        return config
    
class NN_PreprocessLayer(Layer):
    def __init__(self, name="preprocess", **kwargs):
        super(NN_PreprocessLayer, self).__init__(name=name, **kwargs)
        self.preprocess = nasnet.preprocess_input

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(NN_PreprocessLayer, self).get_config()
        return config