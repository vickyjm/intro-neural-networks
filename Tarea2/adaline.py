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

def updateWeights(oldW, delta, rate, features):
	output = []
	for i in range(len(oldW)):
		output.append(oldW[i] + delta*rate*features[i])
	return output

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
	error = 10
	epsilon = 1

	plotOriginalData(x,d)

	while (error > epsilon) and (epoch < 1000):
		error = 0
		for i in range(0,len(x)):
			output = 0
			for j in range(0,len(x[i])):
				output += x[i][j]*w[j]
			
			delta = d[i] - output
			error = error + delta*delta

			w = updateWeights(w,delta,rate,x[i])

		print(error)

		epoch = epoch + 1

	print("Número de épocas: ", epoch)