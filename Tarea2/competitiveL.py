import sys
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

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

	w1 = np.random.randint(low=-500,high=500,size=3)/1000.0
	w2 = np.random.randint(low=-500,high=500,size=3)/1000.0
	rate = 0.1
	epoch = 0
	errors = np.ones(len(x))

	while any(errors) and (epoch < 1000):
		for i in range(0,len(x)):
			output1 = 0
			output2 = 0
			for j in range(0,len(x[i])):
				output1 += x[i][j]*w1[j]
				output2 += x[i][j]*w2[j]



