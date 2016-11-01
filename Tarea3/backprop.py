import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from math import e

def readFile(file):
	data = []
	for line in file:
		tmp = line.split(",")
		data.append([float(tmp[0]),float(tmp[1])])
	return data

# X es una fila del examples
def calcularO(w,x,w0,capa1,capa2):
	output = []
	for i in range(capa2):
		net = w0[i] # w0*x0 y x0 siempre es 1
		for j in range(capa1):     
			net += w[j][i]*x[j]
		output.append(1.0/(1.0 + np.exp(-net*15)))
		
	return output

def calcularOLineal(w,x,w0,capa1,capa2):
	output = []
	for i in range(capa2):
		net = w0[i]
		for j in range(capa1):
			net += w[j][i]*x[j]
		output.append(net)
	return output

def normalizar(matrix,rows,cols) :
	maxAct = -1000
	maxArr = []
	# Buscar el maximo de las columnas
	for j in range(cols):
		for i in range(rows):
			if matrix[i][j] > maxAct : 
				maxAct = matrix[i][j]
		maxArr.append(maxAct)

	# Usando el maximo de cada columna, dividir cada elemento de dicha columna
	# por ese maximo
	for j in range(cols) :
		for i in range(rows):
			matrix[i][j] = matrix[i][j] / maxArr[j]

	return matrix

def backPropagation(examples,rate,nin,nout,nhidden,max_iter):
	np.random.seed(43124)
 
	initW = []  # MATRIZ de pesos entre capa init y hidden
	hidW = []   # MATRIZ de pesos entre capa hidden y out

	# Inicializando pesos entre capa init y capa hidden
	# [[pesosNeurona1],[pesosNeurona2]...] 
	for i in range(0,nin) : 
		initW.append(np.random.randint(low=-500,high=500,size=nhidden)/1000.0)
   # print("Pesos Input-Hidden :",initW)

	# Inicializa el arreglo de umbral entre capa init y hidden
	b0 = np.random.randint(low=-500,high=500,size=nhidden)/1000.0

	#print("Pesos Umbral-Hidden: ",b0)

	# Inicializando pesos entre capa hidden y capa out
	# [[pesosNeurona1],[pesosNeurona2]...]
	for j in range(0,nhidden) :
		hidW.append(np.random.randint(low=-500,high=500,size=nout)/1000.0)
   # print("Pesos Hidden-Output: ",hidW)

	# Inicializa el arreglo de umbral entre capa hidden y out
	b1 = np.random.randint(low=-500,high=500,size=nout)/1000.0
	#print("Pesos Umbral-Output: ",b1)
	aux = 0
	errorArray = []
	iterArray = []
	while (aux < max_iter):
		errorIter = 0
		for ex in examples:

			#Propagate the input forward through the network :
			oHidden = calcularO(initW,ex,b0,nin,nhidden)
			oOut = calcularOLineal(hidW,oHidden,b1,nhidden,nout)
			errorHidden = []
			errorOut = []

			#Propagate the errors backwards :
			#Para cada neurona de salida calcular su error
			for k in range(nout):
				errorOut.append((ex[1]-oOut[k]))
				#errorOut.append(oOut[k]*(1-oOut[k])*(ex[len(ex)-1]-oOut[k]))

			#Para cada neurona intermedia calcular su error
			for j in range(nhidden):
				suma = 0
				for k in range(nout):
					suma += hidW[j][k]*errorOut[k]
				errorHidden.append(oHidden[j]*(1-oHidden[j])*suma)

			# Actualizacion de Pesos : 
			# # Actualizar pesos de las neuronas de input
			for j in range(nhidden):
				for i in range(nin) :
					initW[i][j] += rate*errorHidden[j]*ex[i]
				b0[j] += rate*errorHidden[j]

			# Actualizar pesos de las neuronas entre hidden y out
			for k in range(nout):
				for j in range(nhidden):
					hidW[j][k] += rate*errorOut[k]*oHidden[j]
				b1[k] += rate*errorOut[k]

			err = 0
			#print(errorOut)
			for k in range(len(oOut)):
				err += pow(ex[1]-oOut[k],2)
			errorIter += err/2        

		errorArray.append(errorIter)
		iterArray.append(aux)
		aux = aux + 1
	return initW,b0,hidW,b1,errorArray,iterArray

if __name__ == '__main__':
	trainingFile = open(sys.argv[1],'r')
	#testFile = open(sys.argv[2],'r')

	datos = readFile(trainingFile)
	#testData = readFile(testFile)
	datos = normalizar(datos,71,2)

	trainData, testData = cross_validation.train_test_split(datos,test_size=0.2)

	#testData = normalizar(testData,1000,2)

	#n = [1,2,3]
	#n = [4,6,8]
	#n = [12,20,40]
	#n = [2,4,6,12,20]
	#lr = [0.1,0.01,0.001]
	n = [4]
	lr = [0.1]

	for alpha in lr:
		print("lr="+str(alpha))
		for nHidden in n:
			print("n="+str(nHidden))
			initW,b0,hidW,b1,errArr,iterArr = backPropagation(trainData,alpha,1,1,nHidden,1000)
			#plt.title("ECM datos entrenamiento. n = 4")
			#plt.plot(iterArr,errArr, label="n="+str(nHidden))
			#plt.legend(loc="upper right")
			#plt.ylabel('Error')
			#plt.xlabel('Iteraciones')
			#plt.savefig("n0-1_4_train.png",dpi=200)
			#plt.show()
			
			printLabel = True
			error = 0
			print("Error training:",errArr[499],min(errArr))
			plt.title("Aproximación para n = "+str(nHidden))
			salida = []
			x = []
			aciertos = 0
			
			for p in range(0,len(datos)):
				oHidden = calcularO(initW,datos[p],b0,1,nHidden)
				oOut = calcularOLineal(hidW,oHidden,b1,nHidden,1)

				#if (testData[p][1] == oOut[0]):
					#aciertos += 1
						
				error += pow(datos[p][1]-oOut[0],2)
				salida.append(oOut[0])
				x.append(datos[p][0])

				if printLabel:
					#plt.plot(testData[p][0],testData[p][1],'r^',label="Real")
					#plt.plot(testData[p][0],oOut[0],'bo',label="Aprox")
					plt.plot(datos[p][0],datos[p][1],'r^',label="Real")
					printLabel = False
				else:
					#plt.plot(testData[p][0],testData[p][1],'r^')
					#plt.plot(testData[p][0],oOut[0],'bo')
					plt.plot(datos[p][0],datos[p][1],'r^')


			print("Aciertos: ",aciertos)
			print("Error prueba: ",error/2)

			plt.plot(x,salida,"bo",label="Aprox")
			plt.legend(loc="lower right")
			plt.savefig("n0-1_4_prueba.png",dpi=200)
			plt.show()
