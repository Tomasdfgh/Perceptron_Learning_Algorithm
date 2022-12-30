#Perceptron Learning Rule for Binary Classification
import random
import matplotlib.pyplot as plt
import numpy as np

def linear_model(data,w):
	z = 0
	for i in range(len(data)):
		z += w[i] * data[i]
	return z

def hard_threshold(data):
	if data >= 0:
		return 1
	if data <= 0:
		return -1


def weight_generation(data,min_,max_):
	w = []
	for i in range(len(data[0][0])):
		w.append(random.uniform(min_, max_))
	return w

def data_adaptation(data):
	min_ = min(data[0][0])
	max_ = max(data[0][0])
	for i in range(len(data)):
		l = [1]
		l.extend(data[i][0])
		temp_1 = min(data[i][0])
		temp_2 = max(data[i][0])
		if temp_1 < min_:
			min_ = temp_1
		if temp_2 > max_:
			max_ = temp_2
		data[i][0] = l
	data_trial = data[:int(0.8*len(data))]
	data_test = data[int(0.8*len(data)):]
	return data_trial,data_test,min_,max_

def Perceptron_Algorithm(data,epoch,lr):
	data_trial,data_test,min_,max_ = data_adaptation(data)
	w = weight_generation(data,int(min_),int(max_))

	#---------Graphing---------#

	data_x_t = []
	data_y_t = []
	data_x_f = []
	data_y_f = []
	data_z_t = []
	data_z_f = []

	#---------Graphing for 2D Models---------#

	if len(data[0][0]) == 3:
		for i in range(len(data)):
			if data[i][1] == 1:
				data_x_t.append(data[i][0][1])
				data_y_t.append(data[i][0][2])
			if data[i][1] == -1:
				data_x_f.append(data[i][0][1])
				data_y_f.append(data[i][0][2])

	#---------Graphing for 3D Models---------#

	if len(data[0][0]) == 4:
		for i in range(len(data)):
			if data[i][1] == 1:
				data_x_t.append(data[i][0][1])
				data_y_t.append(data[i][0][2])
				data_z_t.append(data[i][0][3])
			if data[i][1] == -1:
				data_x_f.append(data[i][0][1])
				data_y_f.append(data[i][0][2])
				data_z_f.append(data[i][0][3])

	#---------End of Graphing---------#

	print("________________Training Begins________________")
	count = False
	for i in range(epoch):
		total = 0
		wrong = 0

		for data_iter in data_trial:
			total += 1
			#---------2D Graphing---------#
			if len(data_iter[0]) == 3:

				x = np.linspace(int(min_) - 3,int(max_) + 3,100)
				y = -(w[1]/w[2])*x - (w[0]/w[2])

				plt.cla()
				plt.plot(x,y,"-",data_x_t,data_y_t,"xb",data_x_f,data_y_f,"or")
				plt.title("Binary Classification Using Perceptron Learning Rule")
				plt.ylabel("x_2")
				plt.xlabel("x_1")
				plt.pause(0.0000001)
			#---------3D Graphing---------#
			if len(data_iter[0]) == 4:
				#plt.cla()
				xx, yy = np.meshgrid(range(-100,100), range(-100,100))
				z = -(w[2]/w[3])*yy - (w[1]/w[3])*xx - (w[0]/w[3])
				if count == False:
					fig = plt.figure(figsize = (8,8))
					ax = fig.add_subplot(111,projection = '3d')
				plt.cla()
				ax.scatter(data_x_f,data_y_f,data_z_f,marker = "x",c = "b")
				ax.scatter(data_x_t,data_y_t,data_z_t,marker = "o", c = "r")
				ax.plot_surface(xx, yy, z,alpha = 0.5)
				plt.pause(0.0000001)
				count = True	


	#---------Perceptron Updating Rule---------#
			z = linear_model(data_iter[0],w)
			if z * data_iter[1] <= 0:
				wrong += 1
				for t in range(len(w)):
					w[t] = w[t] + lr*data_iter[1]*data_iter[0][t]


		print("Epoch " + str(i+1) + " has a training accuracy of " + str(float(1 - wrong/total)*100) + "%")
	print("________________Training Terminates________________")

	#---------Algorithm Terminates---------#


	#---------Testing Code---------#
	print(' ')
	total = 0
	correct = 0
	for data_iter in data_test:
		total += 1
		z = linear_model(data_iter[0],w)
		y = hard_threshold(z)
		if y == data_iter[1]:
			correct += 1
	print("Model has a testing accuracy of " + str(float(correct/total)*100) + "%")
	plt.show()