def dW1(w1,w2):
	y = -0.8182 + w1 + 0.8182*w2
	return y

def dW2(w1,w2):
	y = -0.354 + w2 + 0.8182*w1
	return y

w1_old = 0
w2_old = 0
w1_new = 6
w2_new = 6
lr = 1.0
precision = 0.0001

while (abs(w1_new - w1_old) > precision) and (abs(w1_new - w1_old) > precision):
	w1_old = w1_new
	w2_old = w2_new
	w1_new += -lr*dW1(w1_old,w2_old)
	w2_new += -lr*dW2(w1_old,w2_old)

print("Los mínimos locales son: ", w1_new, w2_new)