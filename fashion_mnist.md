

```python
import tensorflow as tf
from tensorflow import keras
```


```python
mnist = keras.datasets.fashion_mnist
```


```python
type(mnist)
```




    tensorflow.python.util.module_wrapper.TFModuleWrapper



#### Version


```python

```


```python
tf._version_()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-34-9cc710eb5e31> in <module>
    ----> 1 tf._version_()
    

    AttributeError: module 'tensorflow' has no attribute '_version_'



```python
tf.__version__
```

### Training and Test Sets


```python
(x_train,y_train), (x_test,y_test)= mnist.load_data()
```

### Exploratory Analysis


```python
x_train.shape
```


```python
y_train.shape
```


```python
x_test.shape
```


```python
y_test.shape
```


```python
x_train[0]
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
np.max(x_train)
```


```python
np.min(x_train)
```


```python
y_train
```


```python
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
```


```python
x_train = x_train/255.0
```

### Building model


```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
```


```python
model= Sequential()
```


```python
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(10,activation='sigmoid'))
```


```python
model.summary()
```

### Model Compilation
#### 1) Loss function
#### 2) Optimizer
#### 3) Metrics


```python
model.compile(optimizer ='adam' , loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
```


```python
model.fit(x_train, y_train, epochs=10)
```


```python
test_loss,test_acc = model.evaluate(x_test,y_test)
print(test_acc)
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
