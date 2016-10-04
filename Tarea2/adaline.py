import sys
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

def plotOriginalData(x,d):
	plt.axes()
	#plt.axis("equal")
	plt.title("Datos originales")
	printLabel1 = True
	printLabel2 = True
	for i in range(len(x)):
		if (d[i] == 0):
			if printLabel1:
				plot = plt.plot(x[i][1],x[i][2],'bo', label="d = 0")
				printLabel1 = False
			else:
				plot = plt.plot(x[i][1],x[i][2],'bo')
		else:
			if printLabel2:
				plot = plt.plot(x[i][1],x[i][2],'r^', label="d = 1")
				printLabel2 = False
			else:
				plot = plt.plot(x[i][1],x[i][2],'r^')
	plt.legend(loc="upper right")
	plt.xlabel("Atributo 1")
	plt.ylabel("Atributo 2")
	plt.show()

def plotClassifier(w,x,data):
	plt.axes()
	plt.title("Adaline")
	printLabel1 = True
	printLabel2 = True

	for xs in data:
		xs = [float(xs[0]),float(xs[1]),float(xs[2])]
		if (xs[2] == -1):
			if printLabel1:
				plot = plt.plot(xs[0],xs[1],'bo', label="d = -1")
				printLabel1 = False
			else:
				plot = plt.plot(xs[0],xs[1],'bo')
		else:
			if printLabel2:
				plot = plt.plot(xs[0],xs[1],'r^', label="d = 1")
				printLabel2 = False
			else:
				plot = plt.plot(xs[0],xs[1],'r^')
	b = w[0]
	w1 = w[1]
	w2 = w[2]
	plt.plot([-2,3],[-b/w2 - w1*(-2)/w2, -b/w2 - w1*(3)/w2])
	plt.ylim([-6,12])
	plt.legend(loc="upper right")
	plt.xlabel("Atributo 1")
	plt.ylabel("Atributo 2")
	#plt.savefig("adaline_1.png",dpi=200)
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
		output.append(oldW[i] + delta*rate*features[i])
	return output

def train(x,d,rate):
	w = [0.492, -0.497,  0.252]
	epoch = 0
	error = 10
	oldError = 0
	epsilon = 0.000005

	while (epoch < 5000):
		oldError = error
		error = 0
		for i in range(0,len(x)):
			output = 0
			for j in range(0,len(x[i])):
				output += x[i][j]*w[j]
			#output = lineal(output)

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
		data.append(line.split(" "))

	#shuffle(data)

	tp = 0
	tn = 0
	fp = 0
	fn = 0
	aciertos = 0

	rate = 0.1

	index = kfold(300,5)

	epoch = 0

	#5-fold validation
	for i in range(0,5):
		d = []
		x = []
		testSamples = []
		testClasses = []

		trainSet = chooseIndex(index,0,i,240,300)
		testSet = chooseIndex(index,1,i,60,300)

		for row in trainSet:
			x.append([1,float(data[row][0]),float(data[row][1])])
			d.append(int(data[row][2]))

		for row in testSet:
			testSamples.append([1,float(data[row][0]),float(data[row][1])])
			testClasses.append(int(data[row][2]))

		w, e = train(x,d,rate)

		plotClassifier(w,x,data)

		epoch += e

		aux = 0
		for sample in testSamples:
			output = 0
			for j in range(len(sample)):
					output += sample[j]*w[j]
			output = sgn(output)
			#print(output)
			error = testClasses[aux] - output

			if (error == 0):  #Bien clasificado
				if (testClasses[aux] == -1):
					tn += 1
				else:
					tp += 1
			else:
				if (testClasses[aux] == -1):
					fp += 1
				else:
					fn += 1

			aux += 1
	
	aciertos = aciertos + tp + tn
	aciertosParcial = aciertos/5
	print("LR: ", rate )
	print("Epocas: ", epoch/5)
	print("Aciertos: ", aciertos)
	print("Aciertos parcial", aciertosParcial)
	print(tp, tn, fp, fn)
	
