# Perceptron Learning Algorithm

## Introduction
This project is a python implementation of the perceptron learning rule; which is a binary linear classification algorithm for supervised learning. This article will contain a brief overview of the perceptron algorithm, a breakdown of the code that I have created, and a discussion of the outcome of the code. Inorder to have a quicker read, feel free to jump directly to the outcome for a discussion of what the code has produced and a visual representation in 2 and 3 dimensions of the outcome.

## Overview
Perceptron learning rule is a modified version of the binary linear classification where instead of a threshold that will result in a 1 for postively classified data points and 0 for negatively classified data points, it is 1 for positive and -1 for negative. Perceptron Learning Algorithm (PLA) then proceeds to iterate through every single data point and determine if the weight vector is currently misclassifying the data point, and update the weight vector in the correct direction. PLA starts with creating a randomly generated weight vector with the same dimensions as each of the data point. Then for each data point, the prediction is determined by dotting the weight vector with the data point. If the value of the prediction multiply by the value of the target (either 1 or -1) is negative, then the model updates itself and that data point is categorized as misclassified. The update is given by

$$w \leftarrow w + t_i*x_i$$

This update guarantees that the direction of the weight vector is being accurately corrected due to the target t. Since if t is negative, it will contain a value of -1 instead of 0 and the update will update accordingly. Hence, that is why t is -1 instead of 0. The full PLA algorithm is therefore

![image](https://user-images.githubusercontent.com/86145397/210059382-eab3076b-9c90-4b43-bb7a-4e8eed9f6086.png)
PLA is guarantee to converge if the data is a convex set. As a result, in order to guarantee convergence, all datasets used during the testing process will be a convex set (ie linearly separable).

## Code
There are two python scripts that make up the entirety of the code (uploaded in this page). The first one is the PLA itself and the second one is a script that creates a random data set that then feeds into the algorithm in order to be process. 
