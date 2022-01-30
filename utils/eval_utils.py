import os
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

from keras.utils.layer_utils import count_params
from sklearn.metrics import accuracy_score

#Below get_flops implementation is taken from "https://github.com/tokusumi/keras-flops/blob/master/notebooks/flops_calculation_tfkeras.ipynb"
def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

def cost_compute(model_path):
    model = load_model(model_path)
    params = float("{0:.2f}".format(model.count_params() / 10 ** 6))
    flops  = float("{0:.2f}".format(get_flops(Model(model.input, model.output), batch_size=1)/ 10 ** 9))
    model_file_size = float("{0:.2f}".format(os.stat(model_path).st_size / 10 ** 6))
    print("FLOPS:", flops, "G")
    print("PARAMETER SIZE:", params, "M")

def evaluate(model, generator):
	#GET TRUE LABELS
	target = generator.classes

	#GET PREDICTIONS
	print("GET PREDICTIONS")
	pred = model.predict(generator, len(generator.filenames)/generator.batch_size, workers=1, verbose=1)
	
	print()
	#EVALUATE ACCURACY
	accuracy = accuracy_score(target, pred.argmax(axis=-1))
	print("#ACCURACY: ", accuracy)

	#EVALUATE MSE
	mse = mean_squared_error(target, pred.argmax(axis=-1))
	print("#MSE: ", mse)

	#EVALUATE MSLE
	msle = mean_squared_log_error(target, pred.argmax(axis=-1))
	print("#MSLE: ", msle)