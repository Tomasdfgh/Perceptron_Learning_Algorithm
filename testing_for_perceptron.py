#Testing File for Perceptron_Learning_Rule_for_Binary_Classification.py
from Perceptron_Learning_Rule_for_Binary_Classification import Perceptron_Algorithm
import matplotlib.pyplot as plt
import random
import numpy as np

def dot(a,w):
	z = 0
	for i in range(len(a)):
		z += a[i] * w[i]
	if z >= 0:
		return 1
	else:
		return -1

def make_up_data(num,dim):
	w = []
	data = []
	data_actual = []
	for i in range(dim+1):
		w.append(random.uniform(-20,20))

	for i in range(num):
		temp = [1]
		temp_2 = []
		temp_3 = []
		tester = []
		target = 0
		for z in range(dim):
			elem = random.randint(-100,100)
			temp.append(elem)
			temp_2.append(elem)
		target = dot(temp,w)
		tester.append(temp)
		tester.append(target)
		data.append(tester)
		temp_3.append(temp_2)
		temp_3.append(target)
		data_actual.append(temp_3)


	#---------Graphing Correct Solution---------#

	data_x_t = []
	data_y_t = []
	data_x_f = []
	data_y_f = []
	data_z_t = []
	data_z_f = []

	for i in range(len(data)):
		#---------2D Graphing---------#
		if len(data[0][0]) == 3:
			if data[i][1] == 1:
				data_x_t.append(data[i][0][1])
				data_y_t.append(data[i][0][2])
			if data[i][1] == -1:
				data_x_f.append(data[i][0][1])
				data_y_f.append(data[i][0][2])
		#---------3D Graphing---------#
		if len(data[0][0]) == 4:
			if data[i][1] == 1:
				data_x_t.append(data[i][0][1])
				data_y_t.append(data[i][0][2])
				data_z_t.append(data[i][0][3])
			if data[i][1] == -1:
				data_x_f.append(data[i][0][1])
				data_y_f.append(data[i][0][2])
				data_z_f.append(data[i][0][3])

	if len(data[0][0]) == 3:
		x = np.linspace(-100,100,100)
		y = -(w[1]/w[2])*x - (w[0]/w[2])
		plt.plot(x,y,"--g",data_x_t,data_y_t,"xb",data_x_f,data_y_f,"or")

	if len(data[0][0]) == 4:
		xx, yy = np.meshgrid(range(-100,100), range(-100,100))
		z = -(w[2]/w[3])*yy - (w[1]/w[3])*xx - (w[0]/w[3])
		fig = plt.figure(figsize = (7,7))
		ax = fig.add_subplot(111,projection = '3d')
		ax.scatter(data_x_f,data_y_f,data_z_f,marker = "x",c = "b")
		ax.scatter(data_x_t,data_y_t,data_z_t,marker = "o", c = "r")
		ax.plot_surface(xx, yy, z,alpha = 0.5)

	plt.show()

	return data_actual

if __name__ == '__main__':
	data = make_up_data(150,3)
	Perceptron_Algorithm(data,5,0.1)