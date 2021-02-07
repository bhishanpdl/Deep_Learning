# Benchmark
I tried the example script given on the official plaidml github.
But it was much slower.
```
   1 : llvm_cpu.0
   2 : opencl_intel_iris(tm)_plus_graphics_650.0
   3 : metal_intel(r)_iris(tm)_plus_graphics_650.0


my tf2 env       : Time taken to run whole notebook: 0 hr 0 min 29 secs
plaidml opencl   : Time taken to run whole notebook: 0 hr 0 min 43 secs
plaidml metal    : Time taken to run whole notebook: 0 hr 0 min 46 secs


Timing inferences...
my tf2         : Ran in 22.587158918380737 seconds
plaidml opencl : Ran in 27.156493186950684 seconds
plaidml metal  : Ran in 25.7920560836792 seconds
```


# Introduction
- [github: plaidml](https://github.com/plaidml/plaidml)

# Installation
- [installation](https://plaidml.github.io/plaidml/docs/install.html)
```bash
# Create new env
yes|conda create -n plaid python=3.8
source activate plaid

# adding new kernel to ipythonsource activate plaid
yes|conda install ipykernel 
python -m ipykernel install --user --name plaid --display-name "Python38(plaid)"
yes|conda install -n plaid  nb_conda_kernels


# special
# https://plaidml.github.io/plaidml/docs/install.html
/Users/poudel/opt/miniconda3/envs/plaid/bin/pip install -U plaidml-keras
plaidml-setup # it gave me 3 options I chose 2nd option with iris gpu


/Users/poudel/opt/miniconda3/envs/plaid/bin/pip install watermark


# conda installations
yes|conda install -n plaid -c conda-forge autopep8  yapf
yes|conda install -n plaid -c conda-forge scikit-plot
yes|conda install -n plaid -c conda-forge seaborn
yes|conda install -n plaid -c conda-forge xgboost lightgbm catboost
yes|conda install -n plaid -c conda-forge plotly_express

```



# Testing code
```python
import numpy as np
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.applications as kapp
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
batch_size = 8
x_train = x_train[:batch_size]
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
model = kapp.VGG19()
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
y = model.predict(x=x_train, batch_size=batch_size)

# Now start the clock and run 10 batches
print("Timing inference...")
start = time.time()
for i in range(10):
    y = model.predict(x=x_train, batch_size=batch_size)
print("Ran in {} seconds".format(time.time() - start))
```