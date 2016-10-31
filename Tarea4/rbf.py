from scipy import *
from scipy.linalg import norm, pinv

def radialFunction(x, center, sigma):
	return exp(-(norm(x - center)**2)/2*(sigma**2))

def calcActivation(x, centers, sigma):
	phi = zeros(len(x), len(centers) + 1, float)

	for c in range(1,len(centers)):
		for xi in range(len(x)):
			phi[xi,c] = radialFunction(x,c,sigma)
	print(phi)
	return phi

def train(x, y, sigma, numCenters):
	#calcular los centros
	index = np.random.randint(low=0,high=len(x),size=numCenters)

	centers = []
	for i in index:
		centers.append(x[i])

	phi = calcActivation(x, centers, sigma)

	#Calcular los pesos usando la pseudoinversa
	w = dot(pinv(phi),y)

	return w

if __name__ == '__main__':
	with open(sys.argv[1]) as fileInput:
		x = []
		d = []
		for line in fileInput:
			tmp = line.split(",")
			x.append(float(tmp[0]))
			d.append(float(tmp[1]))

		sigma = 0.1

		#w = train(x,d,sigma)

		#plotInterpolate()