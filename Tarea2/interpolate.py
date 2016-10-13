import sys
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

def plotOriginalData(data):
	plt.axes()
	#plt.axis("equal")
	plt.title("Datos originales")
	printLabel1 = True
	printLabel2 = True
	for xs in data:
		xs = [float(xs[0]),float(xs[1])]
		plt.plot(xs[0],xs[1],'bo')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

def plotClassifier(w,x,d):
	plt.axes()
	plt.title("Interpolador")
	output = []

	for i in range(len(x)):
		plt.plot(x[i],d[i],'bo')
		output.append(w[0] + w[1]*x[i] + w[2]*pow(x[i],2) + w[3]*pow(x[i],3) + w[4]*pow(x[i],4) + w[5]*pow(x[i],5) + w[6]*pow(x[i],6) + w[7]*pow(x[i],7))
		#output.append(w[0] + w[1]*x[i] + w[2]*pow(x[i],2) + w[3]*pow(x[i],3) + w[4]*pow(x[i],4) + w[5]*pow(x[i],5))
		#output.append(w[0] + w[1]*x[i] + w[2]*pow(x[i],2) + w[3]*pow(x[i],3))
	plt.ylim([-6,12])
	plt.plot(x,output,"g-")
	plt.show()
	plt.pause(0.0001)
	plt.draw()

def kfold(size, k):
	indices = []

	for i in range(size):
		indices.append(i%k)

	indices.sort()

	return indices


def chooseIndex(index, flag, it, size, numSamples):
	res = []
	
	if (flag == 0):
		for i in range(numSamples):
			if (index[i] != it):
				res.append(i)
	else:
		for i in range(numSamples):
			if (index[i] == it):
				res.append(i) 
	return res

def updateWeights(oldW, delta, rate, features):
	output = []
	for i in range(len(oldW)):
		output.append(oldW[i] + delta*rate*features)
	return output

def train(x,d,rate):
	#w = [0.34 ,  0.268,  0.184,  0.415] Cubica
	#w = [-0.445, -0.223, -0.125,  0.243, -0.006, -0.064] #Grado 5
	w = [-0.3  ,  0.133,  0.049, -0.246, -0.221, -0.423,  0.347,  0.269] #Grado 7
	epoch = 0
	grado = 3

	while (epoch < 5000):
		error = 0
		for i in range(0,len(x)):
			output = w[0] + w[1]*x[i] + w[2]*pow(x[i],2) + w[3]*pow(x[i],3) + w[4]*pow(x[i],4) + w[5]*pow(x[i],5) + w[6]*pow(x[i],6) + w[7]*pow(x[i],7)
			#output = w[0] + w[1]*x[i] + w[2]*pow(x[i],2) + w[3]*pow(x[i],3) + w[4]*pow(x[i],4) + w[5]*pow(x[i],5)
			#output = w[0] + w[1]*x[i] + w[2]*pow(x[i],2) + w[3]*pow(x[i],3)
			delta = d[i] - output
			error = error + delta*delta

			w = updateWeights(w,delta,rate,x[i])

		epoch = epoch + 1

	return w, epoch

def sgn(x):
	if (x >= 0):
		return 1
	else:
		return -1

with open(sys.argv[1]) as fileInput:
	d = []
	x = []
	data = []
	for line in fileInput:
		tmp = line.split(" ")
		x.append(float(tmp[0]))
		d.append(float(tmp[1]))

	rate = 0.01
	w, e = train(x,d,rate)

	plotClassifier(w,x,d)