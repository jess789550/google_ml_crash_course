# Introduction to Machine Learning Problem Framing

Problem framing ensures that an ML approach is a good solution to the problem before beginning to work with data and train a model.

---
 
# Understand the problem

To understand the problem, perform the following tasks:
- State the goal for the product you are developing or refactoring.
- Determine whether the goal is best solved using predictive ML, generative AI, or a non-ML solution.
- Verify you have the data required to train a model if you're using a predictive ML approach.

| | Input | Output | Training technique |
|---|---|---|---|
| Predictive ML | Classification; Numerical | Makes prediction e.g. spam email | Uses lots of data to perform specific task |
| Generative AI | Text; Image; Audio; Video; Numerical | Generates output e.g. summary of article, audio, video | Uses unlabeled data for LLM or image generator |

Heuristic = A simple and quickly implemented solution to a problem.

Non-ML vs ML
- Quality
- Cost and maintenance
- A non-ML solution is the benchmark to measure an ML solution against.

Predictive ML and data
- Good predictions needs data that contains features with predictive power
- Abundant dataset
- Consistent and reliable
- Trusted
- Available
- Correct
- Representative

Find feature's predictive power using algorithms:
- Pearson correlation
- Adjusted mutual info
- Shapley value

---

# Framing an ML problem

Tasks:
- Define the ideal outcome and the model's goal.
- Identify the model's output.
- Define success metrics.

Classification
- Binary
- Multiclass single-label
- Multiclass multi-label

Numerical
- Unidimensional regression
- Multidimensional regression

Proxy labels
- Proxy labels substitute for labels that aren't in the dataset
- E.g. is a video useful or not?
- shared or liked --> video is useful
- indirect inference
- Issues: not all users will click like or share

Generation
- most times you'll use a pre-trained generative model to save time
- Distillation: allows the smaller, less resource-intensive model to approximate the performance of the larger model
- Fine-tuning or parameter-efficient tuning: further train the model on a dataset that contains examples of the type of output you want to produce
- Prompt engineering: natural language instructions for how to perform the task or illustrative examples with the desired outputs
- Distillation and fine-tuning update the model's parameters
- prompt engineering helps the model learn how to produce a desired output from the context of the prompt
- test dataset to evaluate a generative model's output against known values
- Generative AI can also be used to implement a predictive ML solution, like classification or regression e.g. LLM

Define the success metrics
- Success metrics differ from the model's evaluation metrics, like accuracy, precision, recall, or AUC.
- E.g. Users spend on average 20 percent more time on the site.
- Outcomes:
	- Not good enough, but continue. 
	- Good enough, and continue
	- Good enough, but can't be made better
	- Not good enough, and never will be

---

# Implementing a model
- start simple
- few features
- can use a pre-trained model if it's suitable for your data (TensorFlow Hub, Kaggle)

Monitoring
- Model deployment: a newly trained model might be worse than the model currently in production so you want alert that your automated deployment has failed
- Training-serving skew: If any of the incoming features used for inference have values that fall outside the distribution range of the data used in training, you'll want to be alerted because it's likely the model will make poor predictions
- Inference server: monitor the RPC server itself and get an alert if it stops providing inferences.
- An RPC (Remote Procedure Call) server is a program on a remote computer that a client can call to execute a function as if it were a local one.
