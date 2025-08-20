# Google Machine Learning Crash Course
https://developers.google.com/machine-learning/crash-course/

---

## Notes

### Intro to ML
- Supervised: regression, clustering
- Unsupervised: clustering

---

### Linear Regression
**Gradient descent** is an iterative process that finds the best weights and bias that minimize the loss.

**Hyperparameters** are variables that control different aspects of training. Three common hyperparameters are:
- Learning rate
- Batch size
- Epochs

In contrast, **parameters** are the variables, like the weights and bias, that are part of the model itself. In other words, hyperparameters are values that you control; parameters are values that the model calculates during training.

**Learning rate** is a floating point number you set that influences how quickly the model converges. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges, but instead bounces around the weights and bias that minimize the loss. The goal is to pick a learning rate that's not too high nor too low so that the model converges quickly.

**Batch size** is a hyperparameter that refers to the number of examples the model processes before updating its weights and bias.

Two common techniques to get the right gradient on average without needing to look at every example in the dataset before updating the weights and bias are stochastic gradient descent and mini-batch stochastic gradient descent:
- **Stochastic gradient descent** uses only a single example (a batch size of one) per iteration.
- **Mini-batch stochastic gradient descent** is a compromise between full-batch and SGD. For
number of data points, the batch size can be any number greater than 1 and less than
- The model chooses the examples included in each batch at random, averages their gradients, and then updates the weights and bias once per iteration.

An **epoch** means that the model has processed every example in the training set once. For example, given a training set with 1,000 examples and a mini-batch size of 100 examples, it will take the model 10 iterations to complete one epoch. In general, more epochs produces a better model, but also takes more time to train.

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
- Logistic regression models use **Log Loss** as the loss function instead of squared loss.
- Applying **regularization** is critical to prevent overfitting.

Most logistic regression models use one of the following two strategies to decrease model complexity:
- **L2 regularization**
- **Early stopping**: Limiting the number of training steps to halt training while loss is still decreasing.

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

**Prediction bias** is the difference between the mean of a model's predictions and the mean of ground-truth labels in the data. 

Prediction bias can be caused by:
- Biases or noise in the data, including biased sampling for the training set
- Too-strong regularization, meaning that the model was oversimplified and lost some necessary complexity
- Bugs in the model training pipeline
- The set of features provided to the model being insufficient for the task

**Multi-class classification** can be treated as an extension of binary classification to more than two classes. For example, in a three-class multi-class classification problem, where you're classifying examples with the labels A, B, and C, you could turn the problem into two separate binary classification problems. First, you might create a binary classifier that categorizes examples using the label A+B and the label C. Then, you could create a second binary classifier that reclassifies the examples that are labeled A+B using the label A and the label B.

---

## Numerical Data

You must determine the best way to represent raw dataset values as trainable values in the feature vector. This process is called feature engineering, and it is a vital part of machine learning. The most common feature engineering techniques are:
- Normalization: Converting numerical values into a standard range.
- Binning (also referred to as bucketing): Converting numerical values into buckets of ranges.

Tip: to get basic stats, use pd.df.describe()
```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

DataFrame.describe(percentiles=None, include=None, exclude=None)

training_df.describe()
```

Normalization provides the following benefits:
- Helps models converge more quickly during training. 
- Helps models infer better predictions. 
- Helps avoid the "NaN trap" when feature values are very high. 
- Helps the model learn appropriate weights for each feature.

**Linear scaling** (more commonly shortened to just scaling) means converting floating-point values from their natural range into a standard range—usually 0 to 1 or -1 to +1. Use when:
- The lower and upper bounds of your data don't change much over time.
- The feature contains few or no outliers, and those outliers aren't extreme.
- The feature is approximately uniformly distributed across its range. That is, a histogram would show roughly even bars for most values.

x' = (x-xmin)(xmax-xmin)

**Z-score scaling**. A Z-score is the number of standard deviations a value is from the mean. Use when data follows a normal distribution.

x' = (x-u)/o

**Log scaling** computes the logarithm of the raw value. Use when:
- power law distribution
- Low values of X have very high values of Y.
- As the values of X increase, the values of Y quickly decrease. Consequently, high values of X have very low values of Y.

x' = ln(x)

**Clipping** is a technique to minimize the influence of extreme outliers. Set a maximum value threshold where anything over that theshold is equal to the maximum.
- If x > max, set x = max
- If x < min, set x = min

**Binning** (also called bucketing) is a feature engineering technique that groups different numerical subranges into bins or buckets. Binning is a good alternative to scaling or clipping when either of the following conditions is met:
- The overall linear relationship between the feature and the label is weak or nonexistent.
- When the feature values are clustered.

**Quantile bucketing** creates bucketing boundaries such that the number of examples in each bucket is exactly or nearly equal. Quantile bucketing mostly hides the outliers.

**Scrubbing** involves removing bad data or cleaning data due to:
- Omitted values	
- Duplicate examples	
- Out-of-range feature values.	
- Bad labels

Qualities of good numerical features
- Clearly named
- Checked or tested before training
- Sensible

When one variable is related to the square, cube, or other power of another variable, it's useful to create a synthetic feature from one of the existing numerical features. It's possible to keep both the linear equation and allow nonlinearity through **polynomial transforms**:

x = x2

y = b + w1x1 + w2x2
