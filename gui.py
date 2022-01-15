import tkinter as tk
from test import main
import argparse

#Input Modes
MODES = ['IMAGE MODE']

#List all model files
#List all model names
BASE_FILES = ['DenseNet201', 'EfficientNetB7', 'InceptionResNetV2', 'ResNet50V2', 'ResNet152V2', 'NASNetLarge', 'Xception', 'InceptionV3', 'DenseNet121', 'EfficientNetB0', 'NASNetMobile', 'MobileNetV2', 'MiniMobileNetV2', 'EnsembleModel']
PROPOSED_FILES = ['MiniMobileNetV2', 'MiniMobileNetV2-KD']
MODEL_FILES = BASE_FILES + PROPOSED_FILES
FONT = 'Helvetica 10 underline'

class ChooseInput:
	def __init__(self, root):
		self.root = root
		self.root.title("PSR Choose Input")

		#Select Mode
		self.l1 = tk.Label(master=root, text="Input Mode", fg="blue", font=FONT)
		self.l1.grid(row=0, column =0, pady = 4)

		self.droptext1 = tk.StringVar()
		self.droptext1.set("IMAGE MODE")
		self.drop1 = tk.OptionMenu(root , self.droptext1 , *MODES, command=self.validate1)
		self.drop1.grid(row=1, column =0, pady = 1, padx = 5)

		#Select Model
		self.l2 = tk.Label(master=root, text="Select Model", fg="blue", font=FONT)
		self.l2.grid(row=0, column =1, pady = 4)

		self.droptext2 = tk.StringVar()
		self.droptext2.set("MiniMobileNetV2-KD")
		self.drop2 = tk.OptionMenu(root , self.droptext2 , *MODEL_FILES, command=self.validate3)
		self.drop2.grid(row=1, column =1, padx = 5)

		#Select Cams
		self.l3 = tk.Label(master=root, text="Select CAM/CAMs", fg="blue", font=FONT)
		self.l3.grid(row=3, column =0, pady = 5, columnspan = 2)

		self.check1 = tk.BooleanVar()  
		self.check2 = tk.BooleanVar()  
		self.check3 = tk.BooleanVar()
		self.check4 = tk.BooleanVar()
		self.check5 = tk.BooleanVar()
		self.check6 = tk.BooleanVar()
			  
		self.cb1 = tk.Checkbutton(root, text = "Grad-Cam", 
			                      variable = self.check1,
			                      onvalue = True,
			                      offvalue = False,
			                      height = 2,
			                      width = 10,
			                      command=self.validate2)
			  
		self.cb2 = tk.Checkbutton(root, text = "Grad-Cam++",
			                      variable = self.check2,
			                      onvalue = True,
			                      offvalue = False,
			                      height = 2,
			                      width = 10,
			                      command=self.validate2)
			  
		self.cb3 = tk.Checkbutton(root, text = "F-ScoreCam",
			                      variable = self.check3,
			                      onvalue = True,
			                      offvalue = False,
			                      height = 2,
			                      width = 10,
			                      command=self.validate2)
		
		self.cb4 = tk.Checkbutton(root, text = "ScoreCam",
			                      variable = self.check4,
			                      onvalue = True,
			                      offvalue = False,
			                      height = 2,
			                      width = 10,
			                      command=self.validate2)

		self.cb5 = tk.Checkbutton(root, text = "CAMERAS",
			                      variable = self.check5,
			                      onvalue = True,
			                      offvalue = False,
			                      height = 2,
			                      width = 10,
			                      command=self.validate2)

		self.cb6 = tk.Checkbutton(root, text = "Guided BP",
			                      variable = self.check6,
			                      onvalue = True,
			                      offvalue = False,
			                      height = 2,
			                      width = 10,
			                      command=self.validate2)
			
		self.cb1.grid(row=4, column =0)
		self.cb2.grid(row=4, column =1)
		self.cb3.grid(row=5, column =0)
		self.cb4.grid(row=5, column =1)
		self.cb5.grid(row=6, column =0)
		self.cb6.grid(row=6, column =1)

		#Choose Images
		self.b1 = tk.Button(master=root, text="SELECT IMAGES", command=self.add_input_path)
		self.b1.grid(row=7, column =0, pady = 2, columnspan = 2)

		self.labletext = tk.StringVar()
		self.labletext.set("0 Selected")
		self.l4 = tk.Label(master=root, textvariable=self.labletext, fg="green")
		self.l4.grid(row=8, column =0, pady = 2, columnspan = 2)

		#Start button
		self.b2 = tk.Button(master=root, text="START",command=self.start)
		self.b2.grid(row=9, column =0, pady = 10, columnspan = 2)
		self.b2.config(state= 'disabled')

		self.input_files = []

	def add_input_path(self):
		#Openfile
		input_path = tk.filedialog.askopenfilenames(title="Select an Image File", initialdir ='./',filetypes=[
                   																			("image", ".jpeg"),
                   																			("image", ".png"),
                   																			("image", ".jpg"),
                   																			])
		self.input_files.extend(input_path)
		self.labletext.set(str(len(self.input_files)) + " Selected")
		
		#Enable START button if atleast one input selected
		if len(self.input_files) > 0:
			self.b2.config( state= 'normal')

	def validate1(self, ele):
		self.b1.config( state= 'normal')
		self.b2.config(state='disabled')

		#Uncheck all checkboxes
		self.check1.set(False)
		self.check2.set(False)
		self.check3.set(False)
		self.check4.set(False)
		self.check5.set(False)
		self.check6.set(False)
		
		#Enable all checkboxes
		self.cb1.config( state= 'normal')
		self.cb2.config( state= 'normal')
		self.cb3.config( state= 'normal')
		self.cb4.config( state= 'normal')
		self.cb5.config( state= 'normal')
		self.cb6.config( state= 'normal')

	def validate2(self):
		is_selected = self.check1.get() or self.check2.get() or self.check3.get() or self.check4.get() or self.check5.get() or self.check6.get()

	def validate3(self, ele):
		#Disable Checkbox in case of Ensenble model
		if self.droptext2.get() == 'EnsembleModel':
			self.cb1.config( state= 'disabled')
			self.cb2.config( state= 'disabled')
			self.cb3.config( state= 'disabled')
			self.cb4.config( state= 'disabled')
			self.cb5.config( state= 'disabled')
			self.cb6.config( state= 'disabled')
		else:
			self.cb1.config( state= 'normal')
			self.cb2.config( state= 'normal')
			self.cb3.config( state= 'normal')
			self.cb4.config( state= 'normal')
			self.cb5.config( state= 'normal')
			self.cb6.config( state= 'normal')

	def close(self):
		#Close GUI
		self.root.destroy()

	def start(self):
		self.close()

		arg_parser = argparse.ArgumentParser(description='Calculate Inference Time', epilog='')
		arg_parser.add_argument("-g", "--enable_gpu", help="Enable GPU", action="store_true", default=False)
		args = arg_parser.parse_args()
		enable_gpu = args.enable_gpu

		#Get arguments
		input_files = self.input_files
		model_name = self.droptext2.get()
		is_gradcam = self.check1.get()
		is_gradcamplus = self.check2.get()
		is_f_scorecam = self.check3.get()
		is_scorecam = self.check4.get()
		is_camerascam = self.check5.get()
		is_guidedbp = self.check6.get()
		
		cams = {'Grad-Cam': is_gradcam, 
				'Grad-Cam++': is_gradcamplus, 
				'Faster ScoreCam': is_f_scorecam,
				'ScoreCam': is_scorecam, 
				'CAMERAS-Cam':is_camerascam,
				'Guided BP':is_guidedbp}

		print("-----------------------------------------------")
		print("[INFO] Input Summary")
		print(" 1.Input Mode: Image Mode")
		print(" 2.Model Name: ", model_name)
		print(" 3.CAMs Selected:", [k for k in cams.keys() if cams[k]])
		print(" 4.Images Selected:", [f.split('/')[-1] for f in input_files])
		print("------------------------------------------------")
		main(model_name, is_gradcam, is_gradcamplus, is_f_scorecam,
						is_scorecam, is_camerascam, is_guidedbp, input_files, enable_gpu)

if __name__ == "__main__":
	print("[INFO] Opening GUI")
	#Initialize GUI
	root = tk.Tk()
	root.resizable(False, False)
	root.geometry("")
	my_gui = ChooseInput(root)
	root.mainloop()