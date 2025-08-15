# Google Machine Learning Crash Course
https://developers.google.com/machine-learning/crash-course/

---

## Notes

### Intro to ML
- Supervised: regression, clustering
- Unsupervised: clustering

---

### Linear Regression
Gradient descent is an iterative process that finds the best weights and bias that minimize the loss.

Hyperparameters are variables that control different aspects of training. Three common hyperparameters are:
- Learning rate
- Batch size
- Epochs

In contrast, parameters are the variables, like the weights and bias, that are part of the model itself. In other words, hyperparameters are values that you control; parameters are values that the model calculates during training.

Learning rate is a floating point number you set that influences how quickly the model converges. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges, but instead bounces around the weights and bias that minimize the loss. The goal is to pick a learning rate that's not too high nor too low so that the model converges quickly.

Batch size is a hyperparameter that refers to the number of examples the model processes before updating its weights and bias.

Two common techniques to get the right gradient on average without needing to look at every example in the dataset before updating the weights and bias are stochastic gradient descent and mini-batch stochastic gradient descent:
- Stochastic gradient descent uses only a single example (a batch size of one) per iteration.
- Mini-batch stochastic gradient descent is a compromise between full-batch and SGD. For
number of data points, the batch size can be any number greater than 1 and less than
- The model chooses the examples included in each batch at random, averages their gradients, and then updates the weights and bias once per iteration.

An epoch means that the model has processed every example in the training set once. For example, given a training set with 1,000 examples and a mini-batch size of 100 examples, it will take the model 10 iterations to complete one epoch. In general, more epochs produces a better model, but also takes more time to train.

---

### Logistic Regression

The standard logistic function, also known as the sigmoid function (sigmoid means "s-shaped"), is used for classification and range from 0 to 1.

f(x) = 1/(1+e^(-x))

Transforming linear output using the sigmoid function:

z = b + w1x1 + w2x2 + ... +wNxN

z is the output of the linear equation, also called the log odds.
b is the bias.
The w values are the model's learned weights.
The x values are the feature values for a particular example.

Logistic regression models are trained using the same process as linear regression models, with two key distinctions:
- Logistic regression models use Log Loss as the loss function instead of squared loss.
- Applying regularization is critical to prevent overfitting.
