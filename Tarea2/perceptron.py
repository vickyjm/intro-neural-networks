import sys
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

def sgn(output):
	if (output > 0):
		return 1
	else:
		return 0

def updateWeights(oldW, error, rate, features):
	output = []
	for i in range(0,len(oldW)):
		output.append(oldW[i]+rate*error*features[i])
	return output

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


with open(sys.argv[1]) as fileInput:
	d = []
	x = []
	data = []
	for line in fileInput:
		data.append(line.split(" "))

	shuffle(data)

	for row in data:
		x.append([1,float(row[0]),float(row[1])])
		d.append(int(row[2]))

	w = np.random.randint(low=-500,high=500,size=3)/1000.0
	rate = 0.1
	epoch = 0
	errors = np.ones(len(x))

	plotOriginalData(x,d)

	while any(errors) and (epoch < 1000):
		for i in range(0,len(x)):
			output = 0
			for j in range(0,len(x[i])):
				output += x[i][j]*w[j]
			output = sgn(output)

			error = d[i] - output
			errors[i] = error

			w = updateWeights(w,error,rate, x[i])

		epoch = epoch + 1

	print("Número de épocas: ", epoch)