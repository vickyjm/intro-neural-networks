from scipy import *
from scipy.linalg import norm, pinv
import numpy as np
import sys
import matplotlib.pyplot as plt

def radialFunction(x, center, sigma):
	return exp(-(norm(x - center)**2)/2*(sigma**2))

def calcActivation(x, centers, sigma):
	phi = np.ones((len(x), len(centers) + 1))

	for c in range(len(centers)):
		for xi in range(len(x)):
			if (xi != 0):
				phi[xi,c] = radialFunction(x[xi],centers[c],sigma)
	#print(phi)
	return phi

def train(x, y, sigma, numCenters):
	#calcular los centros
	index = np.random.randint(low=0,high=len(x),size=numCenters)

	centers = []
	auxCenter = []
	for i in index:
		centers.append(x[i])
		auxCenter.append([x[i],y[i]])
	print(centers)
	phi = calcActivation(x, centers, sigma)

	#Calcular los pesos usando la pseudoinversa
	w = dot(pinv(phi),y)

	return w,auxCenter,phi

def plotOriginalData(x,d):
	plt.axes()
	plt.title("Datos originales")
	for i in range(len(x)):
		plt.plot(x[i],d[i],'bo')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

def plotCenters(x,d,centers):
	plt.axes()
	plt.title("Datos junto a los centros")
	for i in range(len(x)):
		if (i == 0):
			plt.plot(x[i],d[i],'bo',label="Original")
		else:
			plt.plot(x[i],d[i],'bo')

	for i in range(len(centers)):
		if (i == 0):
			plt.plot(centers[i][0],centers[i][1],'r^',label="Centros")
		else:
			plt.plot(centers[i][0],centers[i][1],'r^')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.legend(loc="lower right")
	plt.show()

def plotInterpolate(x,d,w,phi):
	y = dot(phi,w)
	plt.axes()
	plt.title("Interpolación")
	for i in range(len(x)):
		if (i == 0):
			plt.plot(x[i],d[i],'bo',label="Original")
		else:
			plt.plot(x[i],d[i],'bo')

	plt.plot(x,y,'g-',label="Interpolación")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.legend(loc="lower right")
	plt.show()	


if __name__ == '__main__':
	with open(sys.argv[1]) as fileInput:
		x = []
		d = []
		for line in fileInput:
			tmp = line.split(",")
			x.append(float(tmp[0]))
			d.append(float(tmp[1]))

		sigma = 0.001

		#plotOriginalData(x,d)
		w,center,phi = train(x,d,sigma,71)
		#plotCenters(x,d,center)
		plotInterpolate(x,d,w,phi)