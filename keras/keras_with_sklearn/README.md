# Use Keras Deep Learning Models with Scikit-Learn in Python
Reference: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

```python
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

# Use Keras Deep Learning Models with Scikit-Learn in Python
by Jason Brownlee on May 31, 2016 in Deep Learning
Tweet  Share
Last Updated on August 19, 2019

Keras is one of the most popular deep learning libraries in Python for research and development because of its simplicity and ease of use.

The scikit-learn library is the most popular library for general machine learning in Python.

In this post you will discover how you can use deep learning models from Keras with the scikit-learn library in Python.

This will allow you to leverage the power of the scikit-learn library for tasks like model evaluation and model hyper-parameter optimization.

Discover how to develop deep learning models for a range of predictive modeling problems with just a few lines of code in my new book, with 18 step-by-step tutorials and 9 projects.

Let’s get started.

Update: For a larger example of tuning hyperparameters with Keras, see the post:
How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras
Update Oct/2016: Updated examples for Keras 1.1.0 and scikit-learn v0.18.
Update Jan/2017: Fixed a bug in printing the results of the grid search.
Update Mar/2017: Updated example for Keras 2.0.2, TensorFlow 1.0.1 and Theano 0.9.0.
Update Mar/2018: Added alternate link to download the dataset as the original appears to have been taken down.
Use Keras Deep Learning Models with Scikit-Learn in Python
Use Keras Deep Learning Models with Scikit-Learn in Python
Photo by Alan Levine, some rights reserved.

Overview
Keras is a popular library for deep learning in Python, but the focus of the library is deep learning. In fact it strives for minimalism, focusing on only what you need to quickly and simply define and build deep learning models.

The scikit-learn library in Python is built upon the SciPy stack for efficient numerical computation. It is a fully featured library for general machine learning and provides many utilities that are useful in the development of deep learning models. Not least:

Evaluation of models using resampling methods like k-fold cross validation.
Efficient search and evaluation of model hyper-parameters.
The Keras library provides a convenient wrapper for deep learning models to be used as classification or regression estimators in scikit-learn.

In the next sections, we will work through examples of using the KerasClassifier wrapper for a classification neural network created in Keras and used in the scikit-learn library.

The test problem is the Pima Indians onset of diabetes classification dataset. This is a small dataset with all numerical attributes that is easy to work with. Download the dataset and place it in your currently working directly with the name pima-indians-diabetes.csv (update: download from here).

The following examples assume you have successfully installed Keras and scikit-learn.

Need help with Deep Learning in Python?
Take my free 2-week email course and discover MLPs, CNNs and LSTMs (with code).

Click to sign-up now and also get a free PDF Ebook version of the course.

Start Your FREE Mini-Course Now!
Evaluate Deep Learning Models with Cross Validation
The KerasClassifier and KerasRegressor classes in Keras take an argument build_fn which is the name of the function to call to get your model.

You must define a function called whatever you like that defines your model, compiles it and returns it.

In the example, below we define a function create_model() that create a simple multi-layer neural network for the problem.

We pass this function name to the KerasClassifier class by the build_fn argument. We also pass in additional arguments of nb_epoch=150 and batch_size=10. These are automatically bundled up and passed on to the fit() function which is called internally by the KerasClassifier class.

In this example, we use the scikit-learn StratifiedKFold to perform 10-fold stratified cross-validation. This is a resampling technique that can provide a robust estimate of the performance of a machine learning model on unseen data.

We use the scikit-learn function cross_val_score() to evaluate our model using the cross-validation scheme and print the results.

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
 
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

# Grid Search Deep Learning Model Parameters
```python
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
```
