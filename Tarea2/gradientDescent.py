import matplotlib.pyplot as plt

def dW1(w1,w2):
	y = -0.8182 + w1 + 0.8182*w2
	return y

def dW2(w1,w2):
	y = -0.354 + w2 + 0.8182*w1
	return y

def plotClassifier(w1,w2,it):
	plt.axes()
	plt.title("Descenso del Gradiente - w2")
	printLabel = True
	
	plt.plot(it,w1,"bo")	
	plt.plot(it,w1,"-")
	plt.legend(loc="upper right")
	plt.xlabel("Iteraciones")
	plt.ylabel("Pesos")
	plt.savefig("gd1_2.png",dpi=200)
	plt.show()
	plt.pause(0.0001)
	plt.draw()

w1_old = 0
w2_old = 0
w1_new = 6
w2_new = 6
lr = 0.3
precision = 0.0001
w1, w2, it = [], [], []
i = 1

while (abs(w1_new - w1_old) > precision) and (abs(w1_new - w1_old) > precision):
	w1_old = w1_new
	w2_old = w2_new
	w1_new += -lr*dW1(w1_old,w2_old)
	w2_new += -lr*dW2(w1_old,w2_old)
	w1.append(w1_new)
	w2.append(w2_new)
	it.append(i)
	i = i + 1
plotClassifier(w1,w2,it)
print("Los mínimos locales son: ", w1_new, w2_new)