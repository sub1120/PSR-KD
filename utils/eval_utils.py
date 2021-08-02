from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, precision_recall_curve
from tensorflow.keras.models import Model
from keras_flops import get_flops
import numpy as np
import math

#Define Constants
IMAGE_SIZE = (256, 256)
TOTAL_CLASSES = 199
BATCH_SIZE = 4

#Softmax Function
def softmax(arr):
    for i in range(len(arr)):
        ex =np.exp(arr[i])
        arr[i] = ex/np.sum(ex)
    return arr

#Calculate FLOPs Function
def cal_param_flops(models, model_files, results):
    fp = {'Parameters(M)':{}, 'FLOPs(G)':{}, 'ACC(%)':{}}

    for model_name in model_files:
        model = models[model_name]['model']
        flops  = float("{0:.2f}".format(get_flops(Model(model.input, model.output), batch_size=1)/ 10 ** 9))
        params = float("{0:.2f}".format(model.count_params()/10**6))
        acc = float(results['Accuracy(%)'][model_name])
        fp['Parameters(M)'][model_name] = params
        fp['FLOPs(G)'][model_name] = flops
        fp['ACC(%)'][model_name] = acc

    return fp

#Find ROC points to plot
def cal_macro_roc(y_true, y_prob, n_classes):
    fpr = {}
    tpr = {}
  
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
    
    macro_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(macro_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(macro_fpr, fpr[i], tpr[i])
    macro_tpr = mean_tpr/n_classes 
    return macro_fpr, macro_tpr

#Find PR points to plot 
def cal_macro_pr(y_true, y_prob, n_classes):
    pre = {}
    rec = {}
  
    for i in range(n_classes):
        pre[i], rec[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])

    macro_pre = np.unique(np.concatenate([pre[i] for i in range(n_classes)]))
    mean_rec = np.zeros_like(macro_pre)
    for i in range(n_classes):
        mean_rec += np.interp(macro_pre, pre[i], rec[i])
    macro_rec = mean_rec/n_classes
    return macro_rec, macro_pre

#Generate Predictions from model
def get_prediction(model, generator, nb_samples):
    from tqdm.notebook import tqdm
    np.seterr(all='ignore')
    y_true = []
    y_prob = []

    with tqdm(total=nb_samples) as pbar:
        for i in range(math.ceil(nb_samples/BATCH_SIZE)):
            x = generator[i][0]
            y_true.extend(generator[i][1])
            y_prob.extend(model.predict(x))
            pbar.update(len(x))
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_soft_prob = softmax(y_prob)
    y_pred = y_soft_prob.argmax(axis=-1)
    return y_true, y_prob, y_soft_prob, y_pred

#Create Generator
def get_generator(datapath, class_mode, preprocess_input=None, rescale=None):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=rescale)
    datagen = datagen.flow_from_directory(datapath, 
                                            target_size=IMAGE_SIZE, 
                                            batch_size=BATCH_SIZE,
                                            class_mode=class_mode,
                                            shuffle=False)
    nb_samples = len(datagen.filenames)
    return datagen, nb_samples