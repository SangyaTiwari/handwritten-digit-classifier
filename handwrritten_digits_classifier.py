#!/usr/bin/env python
# coding: utf-8

# 
# ### Handwritten Digits Classifier
# 
# In this project, we will build a deep learning model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.
# 
# #### Project Goals:
# - Load and explore the MNIST dataset
# - Build and train deep feedforward neural networks
# - Evaluate and improve model accuracy
# - Visualize results and predictions
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# #### Step 1: Loading the MNIST Dataset
# 
# The MNIST dataset is a collection of 70,000 28x28 grayscale images of handwritten digits, divided into training and testing sets.
# 

# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# #### Step 2: Visualizing a Sample Image
# 
# Let's display the first image in the training dataset to see what the model will be working with.
# 

# In[3]:


plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()


# #### Step 3: Normalizing the Data
# 
# We scale the pixel values from 0–255 to 0–1 to make the training process more efficient and stable.
# 

# In[4]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# #### Step 4: Building the Neural Network Model
# 
# We'll use a simple feedforward neural network with:
# - 1 input layer (flattened from 28x28)
# - 1 hidden layer with 128 neurons and ReLU activation
# - 1 output layer with 10 neurons and softmax activation
# 

# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# ####  Step 5: Compiling the Model
# 
# We use:
# - Optimizer: Adam
# - Loss Function: Sparse Categorical Crossentropy
# - Metric: Accuracy
# 

# In[6]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# #### Step 6: Training the Model
# 
# We now train the model for 5 epochs on the training dataset.
# 

# In[7]:


model.fit(x_train, y_train, epochs=5)


# #### Step 7: Evaluating the Model
# 
# We evaluate the trained model on the test dataset to check its generalization performance.
# 

# In[8]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)


# #### Step 8: Making Predictions
# 
# We'll use the model to predict the label for a test image and compare it to the true label.
# 

# In[9]:


predictions = model.predict(x_test)

plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[0])}, Actual: {y_test[0]}")
plt.axis('off')
plt.show()


# #### Final Summary
# 
# In this project, we built a deep learning model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model was trained on 60,000 labeled images and tested on 10,000 unseen images. 
# 
# Key accomplishments:
# - Preprocessed and normalized grayscale image data.
# - Built a simple neural network using dense layers.
# - Achieved high accuracy on the test set.
# - Visualized predictions and performance.
# 
# This project demonstrates fundamental skills in data preprocessing, model building, evaluation, and visualization using Python-based deep learning tools.
# 

# In[ ]:




