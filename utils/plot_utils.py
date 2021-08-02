#Import plot related modules
from utils.eval_utils import cal_macro_roc, cal_macro_pr, cal_param_flops
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle
import os

#List all model names
DISTILLED_STUDENT_FILES = ['KD-DenseNet121', 'KD-EfficientNetB0', 'KD-NASNetMobile', 'KD-MobileNetV2', 'KD-Custom-CNN']
NORMAL_STUDENT_FILES = ['DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'Custom-CNN']
TEACHER_FILES = ['DenseNet201', 'Xception', 'InceptionResNetV2', 'ResNet152V2', 'EfficientNetB7', 'NASNetLarge', 'EnsembleModel']

#Plot initials
SAVE_PATH = None
MODELS_PATH = None
TEXT_MODE = True
TOTAL_CLASSES = 199
FONT_SIZE = 7.5
DPI = 600

#Models, predictions and results
models = {}
val_preds = {}
test_preds = {}
val_results = {}
test_results = {}

def update_env():
	#Setting matplotlib params
	plt.rcParams.update({'figure.figsize': [5,4]})
	plt.rcParams.update({'figure.dpi': DPI})
	plt.rcParams.update({'font.size': FONT_SIZE})
	plt.rcParams.update({'legend.borderpad': 0.6})
	plt.rcParams.update({'legend.frameon': True})
	plt.rcParams.update({'xtick.direction': 'in'})
	plt.rcParams.update({'ytick.direction': 'in'})

#Load History Function
def load_h(file):
	with open(file + '/' + 'model.history', 'rb') as file_pi:
		his = pickle.load(file_pi)
	return his

#Create {index:classs_name} dictionary
def index_class_dict():
	file_names = os.listdir('ds/test/')
	file_names = sorted(file_names)
	index_class = {}
	for i in range(len(file_names)):
		index_class[i] = file_names[i]

	return index_class

# Confusion plot
def plot_confusion(model_name, mode='VAL'):
	classes = index_class_dict()
	if mode == 'VAL':
		y_true, y_pred = (val_preds[model_name]['y_true'], val_preds[model_name]['y_pred'])
	else:
		y_true, y_pred = (test_preds[model_name]['y_true'], test_preds[model_name]['y_pred'])
	df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index = classes.values(), columns = classes.values())
	plt.figure(figsize = (20, 15))
	sn.heatmap(df_cm, cmap="YlGnBu")
	savefigure(SAVE_PATH + 'CONFUSION MATRIX', model_name + '-CMatrix-' + mode)
	plt.show()

# ROC plot
def plot_roc(model_files, mode='VAL', kd=False):
	legends = []
	min_x, max_x, step_x = 0, 1.1, .1
	min_y, max_y, step_y = .90, 1.01, .01
	grids=':'
	gridw=.5
	lw=.7
	ls='-'

	for model_name in model_files:
		if model_name == 'EnsembleModel': ls='--'
		if mode == 'VAL':
			AUC = val_results['ROC-AUC'][model_name]
			y_true, y_prob = (to_categorical(val_preds[model_name]['y_true']), val_preds[model_name]['y_soft_prob'])
		else:
			AUC = test_results['ROC-AUC'][model_name]
			y_true, y_prob = (to_categorical(test_preds[model_name]['y_true']), test_preds[model_name]['y_soft_prob'])
		
		legends.append(model_name + ' (auc=' + AUC + ')')

		macro_fpr, macro_tpr = cal_macro_roc(y_true, y_prob, TOTAL_CLASSES)
		cond1 = np.logical_and(macro_fpr >= min_x, macro_fpr <= max_x)
		cond2 = np.logical_and(macro_tpr >= min_y, macro_tpr <= max_y)
		cond3 = np.logical_and(cond1, cond2)
		if kd:
			if model_name in DISTILLED_STUDENT_FILES:
				ls = '-'
			elif model_name in NORMAL_STUDENT_FILES:
				ls = ":"
			else:
				ls = "-."
		plt.plot(macro_fpr[cond3], macro_tpr[cond3], lw=lw, ls=ls, color=models[model_name]['color'])
	
	plt.xticks(np.arange(min_x, max_x, step=step_x))
	plt.yticks(np.arange(min_y, max_y, step=step_y))
	plt.grid(b=True, ls=grids, lw=gridw)

	if TEXT_MODE:
		add_text(xlabel='False Positive Rate', ylabel='True Positive Rate', legend=legends, legend_loc='lower right')
	else:
		plt.xticks(np.arange(min_x, max_x, step=step_x), labels=[])
		plt.yticks(np.arange(min_y, max_y, step=step_y), labels=[])
	
	savefigure(SAVE_PATH + 'ROC CURVES/', 'ROC-CURVE-' + mode)
	plt.show()

# PR plot
def plot_pr(model_files, mode='VAL', kd=False):
	legends = []
	min_x, max_x, step_x = 0, 1.1, .1
	min_y, max_y, step_y = 0, 1.1, .1
	grids=':'
	gridw=.5
	lw=.7
	ls='-'

	for model_name in model_files:
		if model_name == 'EnsembleModel': ls='--'
		if mode == 'VAL':
			AUC = val_results['PR-AUC'][model_name]
			y_true, y_prob = (to_categorical(val_preds[model_name]['y_true']), val_preds[model_name]['y_soft_prob'])
		else:
			AUC = test_results['PR-AUC'][model_name]
			y_true, y_prob = (to_categorical(test_preds[model_name]['y_true']), test_preds[model_name]['y_soft_prob'])
			
		legends.append(model_name + ' (auc=' + AUC + ')')

		macro_rec, macro_pre = cal_macro_pr(y_true, y_prob, TOTAL_CLASSES)
		if kd:
			if model_name in DISTILLED_STUDENT_FILES:
				ls = '-'
			elif model_name in NORMAL_STUDENT_FILES:
				ls = ":"
			else:
				ls = "-."
		plt.plot(macro_rec, macro_pre, lw=lw, ls=ls, color=models[model_name]['color'])

	plt.xticks(np.arange(min_x, max_x, step=step_x))
	plt.yticks(np.arange(min_y, max_y, step=step_y))
	plt.grid(b=True, ls=grids, lw=gridw)
		
	if TEXT_MODE:    
		add_text(xlabel='Recall', ylabel='Precision', legend=legends, legend_loc='lower left')
	else:
		plt.xticks(np.arange(min_x, max_x, step=step_x), labels=[])
		plt.yticks(np.arange(min_y, max_y, step=step_y), labels=[])
	
	savefigure(SAVE_PATH + 'PR CURVES/', 'PR-CURVE-' + mode)
	plt.show()

# ACC plot
def plot_acc(model_files, kd=False):
	legends = []
	min_x, max_x, step_x = 0, 31, 2
	min_y, max_y, step_y = 0, 110, 10
	epochs = np.arange(1, 31, step=1)
	grids='-'
	gridw=.2

	for model_name in model_files:
		his = load_h(MODELS_PATH + model_name)
		if kd:
			train_acc =[acc*100 for acc in his['categorical_accuracy']]
		else:
			train_acc =[acc*100 for acc in his['accuracy']]
		legends.append(model_name+'-TRAIN')
		plt.plot(epochs, train_acc,  marker='o', markersize=2.5, ls='-', lw=.5, color=models[model_name]['color'], fillstyle='none', markeredgewidth=.5)

	for model_name in model_files:
		his = load_h(MODELS_PATH + model_name)
		if kd:
			val_acc =[acc*100 for acc in his['val_categorical_accuracy']]
		else:
			val_acc =[acc*100 for acc in his['val_accuracy']]
		legends.append(model_name+'-VAL')
		plt.plot(epochs, val_acc ,  marker='^', markersize=2.5, ls='--', lw=.5, color=models[model_name]['color'], fillstyle='none', markeredgewidth=.5)

	plt.xticks(np.arange(min_x, max_x, step=step_x))
	plt.yticks(np.arange(min_y, max_y, step=step_y))
	plt.grid(b=True, ls=grids, lw=gridw)

	if TEXT_MODE:    
		add_text(xlabel='Epoch No.', ylabel='Accuracy(%)', legend=legends, legend_loc='lower right', col=2)
	else:
		plt.xticks(np.arange(min_x, max_x, step=step_x), labels=[])
		plt.yticks(np.arange(min_y, max_y, step=step_y), labels=[])

	savefigure(SAVE_PATH + 'ACCURACY LOSS CURVES/', 'ACCURACY-CURVE')
	plt.show()

#Plot loss
def plot_loss(model_files, kd=False):
	legends = []
	min_x, max_x, step_x = 0, 31, 2
	min_y, max_y, step_y = 0, 6, .5
	epochs = np.arange(1, 31, step=1)
	grids='-'
	gridw=.2

	for model_name in model_files:
		his = load_h(MODELS_PATH + model_name)
		if kd:
			train_loss = his['combined_loss']
		else:
			train_loss = his['loss']
		legends.append(model_name+'-TRAIN')
		plt.plot(epochs, train_loss,  marker='o', markersize=2.5, ls='-', lw=.5, color=models[model_name]['color'], fillstyle='none', markeredgewidth=.5)

	for model_name in model_files:
		his = load_h(MODELS_PATH + model_name)
		if kd:
			val_loss = his['student_loss']
		else:
			val_loss = his['val_loss']
		legends.append(model_name+'-VAL')
		plt.plot(epochs, val_loss,  marker='^', markersize=2.5, ls='--', lw=.5, color=models[model_name]['color'], fillstyle='none', markeredgewidth=.5)

	plt.xticks(np.arange(min_x, max_x, step=step_x))
	plt.yticks(np.arange(min_y, max_y, step=step_y))
	plt.grid(b=True, ls=grids, lw=gridw)

	if TEXT_MODE:    
		add_text(xlabel='Epoch No.', ylabel='Loss', legend=legends, legend_loc='upper right',col=2)
	else:
		plt.xticks(np.arange(min_x, max_x, step=step_x), labels=[])
		plt.yticks(np.arange(min_y, max_y, step=step_y), labels=[])

	savefigure(SAVE_PATH + 'ACCURACY LOSS CURVES/', 'LOSS-CURVE')
	plt.show()

#Plot FLOPS, Accuracy, Parameters
def plot_bubble(model_files, fp, mode='VAL'):
	min_x, max_x, step_x = 0, 40, 8
	min_y, max_y, step_y = 50, 101, 5
	grids=':'
	gridw=.5

	df = pd.DataFrame(fp)
	colors = ['chocolate','olive', 'c','m', 'royalblue' , 									#KD-Student colors
	'chocolate','olive', 'c','m', 'royalblue', 			 									#Normal Student colors
	'tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:brown', 'navy'		#Teacher colors
	]
	hatchs = ['/','/','/','/','/','','','','','','','','','','','','']
	bubble_chart_kd = pd.DataFrame({'FLOPs': df['FLOPs(G)'][:5],
								 'ACC':df['ACC(%)'][:5], 
								 'bubble_size': df['Parameters(M)'][:5]*20,
								 'center_size': df['Parameters(M)'][:5]*.05,
								 'facecolor':colors[:5]})

	bubble_chart = pd.DataFrame({'FLOPs': df['FLOPs(G)'][5:],
								 'ACC':df['ACC(%)'][5:], 
								 'bubble_size': df['Parameters(M)'][5:]*20,
								 'center_size': df['Parameters(M)'][5:]*.05,
								 'facecolor':colors[5:]})

	plt.rcParams.update({'font.size': 6})

	plt.scatter('FLOPs', 'ACC', s='bubble_size', data=bubble_chart_kd, alpha=.5, linewidth=0, hatch = '////', facecolor='facecolor', edgecolor='white')
	plt.scatter('FLOPs', 'ACC', s='bubble_size', data=bubble_chart, alpha=.5, linewidth=0, facecolor='facecolor')
	plt.xticks(np.arange(min_x, max_x, step=step_x))
	plt.yticks(np.arange(min_y, max_y, step=step_y))
	plt.grid(True, ls=grids, lw=gridw)

	if TEXT_MODE:
		add_text(xlabel='FLOPs(G)', ylabel='Accuracy(%)', legend=[], legend_loc='upper right',col=1)
	else:
		plt.xticks(np.arange(min_x, max_x, step=step_x), labels=[])
		plt.yticks(np.arange(min_y, max_y, step=step_y), labels=[])
	
	for model_name in model_files:
		pos_x = fp['FLOPs(G)'][model_name]
		pos_y = fp['ACC(%)'][model_name]
		if model_name in TEACHER_FILES:
			plt.annotate(model_name + ' ' + str(fp['FLOPs(G)'][model_name]) + 'G', (pos_x, pos_y),  (pos_x - 3, pos_y - 2))
		elif model_name in NORMAL_STUDENT_FILES:
			plt.annotate(model_name + ' ' + str(fp['FLOPs(G)'][model_name]) + 'G', (pos_x, pos_y),  (pos_x - 1, pos_y + 1))
		else:
			plt.annotate(model_name + '\n' + str(fp['FLOPs(G)'][model_name]) + 'G', (pos_x, pos_y),  (pos_x - 1, pos_y + 1))
		
	savefigure('figures/', 'FLOPs-' + mode)
	plt.show()
	plt.rcParams.update({'font.size': 7.5})

#All Text Work in plots
def add_text(xlabel, ylabel, legend, legend_loc, col=1):
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(legend, loc=legend_loc, ncol=col)

#Save Figure Function
def savefigure(directory, fig_name):
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory + '/' + fig_name + '.tiff', bbox_inches='tight', dpi=DPI, format='tiff')