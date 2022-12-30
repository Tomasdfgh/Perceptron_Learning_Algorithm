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
There are two python scripts that make up the entirety of the code (uploaded in this page). The first one is the PLA itself and the second one is a script that creates a random data set that then feeds into the algorithm in order to be process. In the dataset creation script, called testing_for_perceptron.py, user can choose how many data points in the set and also the dimensions of each data point. If the data has a dimension of either 2 or 3, then there will be a visual representation of what the data will look like in the form of a 2D or 3D graph. testing_for_perceptron.py will create a randomized weight vector and generate the requested number of data points and a target for each data point depending on where they are in relation to the weight vector. This will serve as the "correct answer" for the PLA to figure out. Once the data set has been created, only the data points and its target will be fed into the PLA and not the correct weight vector. The job of the PLA will now be finding out what that weight vector is in order to correctly classify the data.

Once PLA (which the script is called Perceptron_Learning_Rule_for_Binary_Classification.py) has recieved the dataset, it will immediately begin to preprocess the data before running the algorithm. The dataset is broken into a training set and a test set where training set makes up 80 percent of the data and test set makes up the other 20. In addition, Since we know that linear model is given as

$$ z = w^Tx + b $$

where w is the weight vector, x is a data point, and b is the bias constant. In order to remove the b constant, it can be intergrated into the w and x vector by adding an extra dimension to w called $w_0$ where $w_0 = b$. and an extra x dimension called $x_0$ where $x_0 = 1$. Such that
![image](https://user-images.githubusercontent.com/86145397/210060884-e3bf7148-c987-4323-8e80-ecc562301396.png)
Therefore, the bias is removed from the linear model without compromising its integrity and the model is reduced to

$$ z = w^Tx $$

Once preprocessing is done, a weight vector will be generated at random and will begin to update as the algorithm proceeds. The algorithm will be iterating through every single data point in the training set, or an epoch, and be evaluated on their accuracy before repeating itself. The number of epochs trained is a hyperparameter that the user can decide. Once training is done, the model will begin to test the remaining 20 percent of the data in the test set, and a test accuracy will be produce.

## Result

The PLA implementation has been generalized so that it can successfully run on datasets with any positive dimensions, and will converge as long as the data set is a convex set; however, only 2D and 3D datasets are created inorder to have a visual component. With that being said, this script can still be used to run datasets with any dimensions.

### 2D Trial
The dataset created from the testing_for_perceptron.py can be viewed below

![Correct_Answer_2D](https://user-images.githubusercontent.com/86145397/210061862-73bb3701-3676-43ed-8858-33fa17732c1f.png)

This dataset contains 150 data points, each with a dimension of 2. The weight vector seen in this photo is the correct weight vector and now the PLA will attempt to converge to the most accurate weight vector.

![2d_result_gif](https://user-images.githubusercontent.com/86145397/210062026-21ddd8d9-d9e8-455b-a6d7-e516b91388e8.gif)

As seen in this GIF, the weight vector is slowly converging to the correct solution over the 5 epochs. The accuracy result of this example is given below

![image](https://user-images.githubusercontent.com/86145397/210062385-6f485459-1692-40eb-b1c3-7bebf1215c5e.png)

where the model has an overall accuracy of 96.67 percent.

### 3D Trial

The dataset created from the testing_for_perceptron.py can be viewed below

![3D_CorrectSolution](https://user-images.githubusercontent.com/86145397/210062492-11d58d6f-e2d5-45b8-828e-23037b8795a8.png)

Ofcourse, having 3 dimensions, the weight vector now exists as a plane and each data point also exists on a 3D plane. This data set also contains 150 points, each with a dimension of 3. The weight plane is the correct weight plane and now the PLA will attempt to converge to the most accurate weight plane.

![3d_result_gif](https://user-images.githubusercontent.com/86145397/210062666-b738b137-2a82-45b4-8e45-6f235e5e9ac5.gif)

As seen in this GIF, the weight plane is slowly converging to the correct solution over the 5 epochs. After 5 epochs, the accuracy of this example can be seen below

![image](https://user-images.githubusercontent.com/86145397/210062911-474ecd65-becf-4fbb-99d3-5d4d7b2df82b.png)

This model has a testing accuracy of 93.33 percent.


