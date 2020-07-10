#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


mnist = keras.datasets.fashion_mnist


# In[4]:


type(mnist)


# #### Version

# In[5]:


# Tensorflow version


# In[6]:


#------------------


# In[7]:


tf.__version__


# ### Training and Test Sets

# In[8]:


(x_train,y_train), (x_test,y_test)= mnist.load_data()


# ### Exploratory Analysis

# In[9]:


x_train.shape


# In[10]:


y_train.shape


# In[11]:


x_test.shape


# In[12]:


y_test.shape


# In[13]:


x_train[0]


# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


np.max(x_train)


# In[16]:


np.min(x_train)


# In[17]:


y_train


# In[18]:


plt.figure()
plt.imshow(x_train[0])
plt.colorbar()


# In[19]:


x_train = x_train/255.0


# ### Building model

# In[28]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense


# In[29]:


model= Sequential()


# In[30]:


model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[31]:


model.summary()


# ### Model Compilation
# #### 1) Loss function
# #### 2) Optimizer
# #### 3) Metrics

# In[32]:


model.compile(optimizer ='adam' , loss='sparse_categorical_crossentropy', metrics= ['accuracy'])


# In[41]:


history = model.fit(x_train, y_train, epochs=10, batch_size=10,validation_split=0.2)


# In[48]:


test_loss,test_acc = model.evaluate(x_test,y_test)
print(test_acc)


# #### pip install mlxtend

# In[27]:


help(model)


# #### Learning Curve - We want to plot loss and accuracy function

# In[44]:


history.history


# In[46]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[47]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


# In[50]:


from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(x_test)


# In[51]:


accuracy_score(y_test,y_pred)


# ### Confusion Matrix - mlxtend

# In[62]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib


# In[68]:


font = {
    'family':'Times New Roman',
    'size':12
}
matplotlib.rc('font',**font)
mat = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=mat,figsize=(8,8),class_names= class_names,show_normed= True)


# In[57]:


class_names =["top","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




