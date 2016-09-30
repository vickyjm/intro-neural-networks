import sys

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


with open(sys.argv[1]) as fileInput:
	d = []
	x = []
	for line in fileInput:
		row = line.split(" ")
		x.append([1,float(row[0]),float(row[1])])
		d.append(int(row[2]))

	w = [0,0,0]
	rate = 0.1
	epoch = 0

	while (epoch < 1000):
		for i in range(0,len(x)):
			output = 0
			for j in range(0,len(x[i])):
				output += x[i][j]*w[j]
			output = sgn(output)

			error = d[i] - output

			w = updateWeights(w,error,rate, x[i])

		epoch = epoch + 1