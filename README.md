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

Most logistic regression models use one of the following two strategies to decrease model complexity:
- L2 regularization
- Early stopping: Limiting the number of training steps to halt training while loss is still decreasing.

---

### Classification

Binary classification means interpreting the logstic regression probability value as one category or another e.g. spam/not spam. This requires a threshold.

A confusion matrix is made up of TP, FP, TN, and FN.

When the total of actual positives is not close to the total of actual negatives, the dataset is imbalanced. An instance of an imbalanced dataset might be a set of thousands of photos of clouds, where the rare cloud type you are interested in, say, volutus clouds, only appears a few times.

Accuracy is the proportion of all classifications that were correct, whether positive or negative.

The true positive rate (TPR), or the proportion of all actual positives that were classified correctly as positives, is also known as recall.

The false positive rate (FPR) is the proportion of all actual negatives that were classified incorrectly as positives, also known as the probability of false alarm.

Precision is the proportion of all the model's positive classifications that are actually positive.

| Metric	| Guidance |
| --------- | ------------------- |
| Accuracy	| Use as a rough indicator of model training progress/convergence for balanced datasets. For model performance, use only in combination with other metrics. Avoid for imbalanced datasets. Consider using another metric. | 
| Recall (True positive rate) |	Use when false negatives are more expensive than false positives. | 
| False positive rate	| Use when false positives are more expensive than false negatives. | 
| Precision	| Use when it's very important for positive predictions to be accurate. | 

The ROC curve is a visual representation of model performance across all thresholds. The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold (in practice, at selected intervals), then graphing TPR over FPR. A perfect model, which at some threshold has a TPR of 1.0 and a FPR of 0.0, can be represented by either a point at (0, 1).

The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative. The perfect model above, containing a square with sides of length 1, has an area under the curve (AUC) of 1.0. For a binary classifier, a model that does exactly as well as random guesses or coin flips has a ROC that is a diagonal line from (0,0) to (1,1). The AUC is 0.5, representing a 50% probability of correctly ranking a random positive and negative example. AUC is a useful measure for comparing the performance of two different models, as long as the dataset is roughly balanced.

Prediction bias is the difference between the mean of a model's predictions and the mean of ground-truth labels in the data. 

Prediction bias can be caused by:
- Biases or noise in the data, including biased sampling for the training set
- Too-strong regularization, meaning that the model was oversimplified and lost some necessary complexity
- Bugs in the model training pipeline
- The set of features provided to the model being insufficient for the task

Multi-class classification can be treated as an extension of binary classification to more than two classes. For example, in a three-class multi-class classification problem, where you're classifying examples with the labels A, B, and C, you could turn the problem into two separate binary classification problems. First, you might create a binary classifier that categorizes examples using the label A+B and the label C. Then, you could create a second binary classifier that reclassifies the examples that are labeled A+B using the label A and the label B.
