from scipy import *
from scipy.linalg import norm, pinv
import numpy as np
import sys
import matplotlib.pyplot as plt

def som(epochs, etha, x, d, nn):
	centers = []

	# Una capa con nn neuronas
	w = np.random.randint(low=0,high=len(x),size=nn)/1000.0

	for i in range(epochs):
		#desordenar los datos

		for j in range(len(x)):
			# calculo la distancia entre la entrada y cada vector de pesos
			

if __name__ == '__main__':
	with open(sys.argv[1]) as fileInput:
		x = []
		d = []
		for line in fileInput:
			tmp = line.split(" ")
			x.append(float(tmp[0]))
			d.append(float(tmp[1]))

		etha = 0.001

		#plotOriginalData(x,d)
		w,center,phi = train(x,d,sigma,71)
		#plotCenters(x,d,center)
		plotInterpolate(x,d,w,phi)