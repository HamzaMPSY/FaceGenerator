from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import os
import sys
from os import path
import torch
from torch import nn
import numpy as np

FORM_CLASS,_= loadUiType(path.join(path.dirname(__file__),"main.ui"))

model = None

class Generator(nn.Module):
	'''
	Generator Class
	Values:
	z_dim: the dimension of the noise vector, a scalar
	im_chan: the number of channels in the images, fitted for the dataset used, a scalar
	hidden_dim: the inner dimension, a List
	'''
	def __init__(self, z_dim=100, im_chan=3, hidden_dim=64):
		super(Generator, self).__init__()
		self.z_dim = z_dim
		# Build the neural network
		self.gen = nn.Sequential(
			self.make_gen_block(z_dim, hidden_dim * 8,kernel_size=4, stride=1),
			self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2,padding = 1),
			self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2,padding = 1),
			self.make_gen_block(hidden_dim * 2, hidden_dim , kernel_size=4, stride=2,padding = 1),
			self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding = 1,final_layer=True)
		)

	def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2,padding = 0, final_layer=False):
		'''
		Function to return a sequence of operations corresponding to a generator block of DCGAN, 
		corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
		Parameters:
		input_channels: how many channels the input feature representation has
		output_channels: how many channels the output feature representation should have
		kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
		stride: the stride of the convolution
		final_layer: a boolean, true if it is the final layer and false otherwise 
		(affects activation and batchnorm)
		'''
		# Build the neural block
		if not final_layer:
			return nn.Sequential(
				nn.ConvTranspose2d(in_channels=input_channels,out_channels=output_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
				nn.BatchNorm2d(num_features=output_channels),
				nn.ReLU(True)    
			)
		# Final Layer
		return nn.Sequential(
			nn.ConvTranspose2d(in_channels=input_channels,out_channels=output_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias= False),
			nn.Tanh()
		)
	def unsqueeze_noise(self, noise):
		'''
		Function for completing a forward pass of the generator: Given a noise tensor, 
		returns a copy of that noise with width and height = 1 and channels = z_dim.
		Parameters:
		noise: a noise tensor with dimensions (n_samples, z_dim)
		'''
		return noise.view(len(noise), self.z_dim, 1, 1)

	def forward(self, noise):
		'''
		Function for completing a forward pass of the generator: Given a noise tensor, 
		returns generated images.
		Parameters:
		noise: a noise tensor with dimensions (n_samples, z_dim)
		'''
		x = self.unsqueeze_noise(noise)
		return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
	'''
	Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
	creates a tensor of that shape filled with random numbers from the normal distribution.
	Parameters:
	n_samples: the number of samples to generate, a scalar
	z_dim: the dimension of the noise vector, a scalar
	device: the device type
	'''
	return torch.randn(n_samples, z_dim, device=device)


def loadModel():
	global model
	print(torch.__version__)
	model = torch.load('gangen.pt',map_location=torch.device('cpu')).double()
	print('Model loaded!')

class MainApp(QMainWindow,FORM_CLASS):
	"""docstring for MainApp"""
	def __init__(self, arg=None):
		super(MainApp, self).__init__(arg)
		QMainWindow.__init__(self)
		self.setupUi(self)
		self.handelUI()
		self.handelButton()
		self.handelSliders()
		self.generateRandomFace()

	def handelUI(self):
		self.setWindowTitle('Face Generator')
		self.setFixedSize(788,512)

	def handelButton(self):
		self.pushButton.clicked.connect(self.generateRandomFace)

	def handelSliders(self):
		sliders = self.findChildren(QSlider)
		for slider in sliders:
			slider.valueChanged.connect(self.generateFace)

	def generateRandomFace(self):
		features = get_noise(1,100)
		sliders = self.findChildren(QSlider)
		for v,slider in zip(features[0],sliders):
			slider.setValue((v+1)*50)
		self.draw(features.double())

	def generateFace(self):
		features = [[]]
		sliders = self.findChildren(QSlider)
		for i,slider in enumerate(sliders):
			features[0].append( -1 + slider.value()/50)
		features = torch.DoubleTensor(features,device = 'cpu').double()
		# features = get_noise(1,100)
		self.draw(features)

	def draw(self,features):
		global model
		image = model(features)[0]
		image = (image+1) / 2
		image = image.detach().numpy()
		image = np.transpose(image, (1, 2, 0))
		image = np.round(image*255).astype(np.uint8)
		image = np.require(image, np.uint8, 'C')
		qImg = QImage(image, 64, 64, QImage.Format_RGB888)
		pix =  QPixmap(qImg)
		self.label.setPixmap(pix.scaled(self.label.size()))

def main():
	loadModel() 
	app = QApplication(sys.argv)
	window = MainApp()
	window.show()
	app.exec_()

if __name__ == '__main__':
	main()