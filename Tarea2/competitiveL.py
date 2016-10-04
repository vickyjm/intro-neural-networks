import sys
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

def kfold(size, k):
	indices = []

	for i in range(size):
		indices.append(i%k)

	indices.sort()

	return indices

def plotClassifier(w1,w2,data):
	plt.axes()
	plt.title("Aprendizaje con reforzamiento")
	printLabel1 = True
	printLabel2 = True

	for xs in data:
		xs = [float(xs[0]),float(xs[1]),float(xs[2])]
		if (xs[2] == 0):
			if printLabel1:
				plot = plt.plot(xs[0],xs[1],'bo', label="d = 0")
				printLabel1 = False
			else:
				plot = plt.plot(xs[0],xs[1],'bo')
		else:
			if printLabel2:
				plot = plt.plot(xs[0],xs[1],'r^', label="d = 1")
				printLabel2 = False
			else:
				plot = plt.plot(xs[0],xs[1],'r^')
	b = w1[0]
	tmpw1 = w1[1]
	tmpw2 = w1[2]
	plt.plot([-2,3],[-b/tmpw2 - tmpw1*(-2)/tmpw2, -b/tmpw2 - tmpw1*(3)/tmpw2])
	b = w2[0]
	tmpw1 = w2[1]
	tmpw2 = w2[2]
	plt.plot([-2,3],[-b/tmpw2 - tmpw1*(-2)/tmpw2, -b/tmpw2 - tmpw1*(3)/tmpw2])
	plt.ylim([-6,12])
	plt.legend(loc="upper right")
	plt.xlabel("Atributo 1")
	plt.ylabel("Atributo 2")
	plt.savefig("cl.png",dpi=200)
	plt.show()
	plt.pause(0.0001)
	plt.draw()


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

def train(x,d,rate):
	w1 = np.array([0.492, -0.497, 0.497], dtype=np.float)
	w2 = np.array([0.337,  0.426, 0.426], dtype=np.float)
	epoch = 0
	errors = np.ones(len(x))

	while any(errors) and (epoch < 10000):
		for i in range(0,len(x)):
			output1 = 0
			output2 = 0
			for j in range(0,len(x[i])):
				output1 += x[i][j]*w1[j]
				output2 += x[i][j]*w2[j]

			if (output1 >= output2): 	#Se activó la neurona 1
				if (d[i] == 1):			#Clasificó mal
					for j in range(len(w1)):
						#print(rate*x[i][j])
						w1[j] = w1[j] - rate*x[i][j]
						w2[j] = w2[j] + rate*x[i][j]
				outClass = 0
			else: 						#Se activó la neurona 2
				if (d[i] == 0):			#Clasificó mal
					for j in range(len(w1)):
						w2[j] = w2[j] - rate*x[i][j]
						w1[j] = w1[j] + rate*x[i][j]
				outClass = 1
			errors[i] = d[i] - outClass

		epoch = epoch + 1

	return w1, w2, epoch


with open(sys.argv[1]) as fileInput:
	d = []
	x = []
	data = []
	for line in fileInput:
		data.append(line.split(" "))

	shuffle(data)

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

		w1, w2, e = train(x,d,rate)

		#plotClassifier(w1,w2,data)

		epoch += e

		aux = 0
		for sample in testSamples:
			output1 = 0
			output2 = 0
			for j in range(len(sample)):
					output1 += sample[j]*w1[j]
					output2 += sample[j]*w2[j]
			
			if (output1 >= output2): 	#Se activó la neurona 1
				if (testClasses[aux] == 1): #Clasificó mal
					fn += 1
				else:						#Clasificó bien
					tn += 1
			else:
				if (testClasses[aux] == 0): #Clasificó mal
					fp += 1
				else: 
					tp += 1

			aux += 1
	
	aciertos = aciertos + tp + tn
	aciertosParcial = aciertos/5
	print("LR: ", rate )
	print("Epocas: ", epoch/5)
	print("Aciertos: ", aciertos)
	print("Aciertos parcial", aciertosParcial)
	print(tp, tn, fp, fn)