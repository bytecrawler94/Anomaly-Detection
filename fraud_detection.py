# -*- coding: utf-8 -*-
"""fraud_detection.ipynb
we use TensorFlow 1.2 and Keras 2.0.4. 
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                                roc_curve, recall_score, classification_report, f1_score,
                                precision_recall_fscore_support, precision_score)

# %matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# Load the data
# The dataset can be downloaded from [Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud). 
df = pd.read_csv("creditcard.csv")

# Autoencoders
# Prepare the data
# Drop the time column and use StandardScaler on the Amount


data = df.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# Split the data into training and testing sets

X_train, X_test = train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values

# Build the model

# The Autoencoder uses 4 fully connected layers with 15, 7, 7 and 29 neurons respectively.  
# First two layers are for encoder, the last two are for the decoder. 
#  L1 regularization will be used while training:


input_dim = X_train.shape[1]
encoding_dim = 15
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Train Data

nb_epoch = 20
batch_size = 32

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                                verbose=0,
                                save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

autoencoder.save('./model.h5')

autoencoder = load_model('model.h5')

# Evaluate the model

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

# Reconstruction error on training data
predictions = autoencoder.predict(X_test)
print(predictions[0])

mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
## Precision vs Recall
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

bprecision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(precision,  'b', label='precision')
print(precision)
plt.xlabel('step')
plt.ylabel('precision')
plt.show()
# Recall vs Threshold

plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()

# Prediction

threshold = 2.8
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# Confusion matrix

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# accuracy: (tp + tn) / (p + n)
# accuracy = accuracy_score(error_df.true_class, y_pred)
# print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(error_df.true_class, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(error_df.true_class, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(error_df.true_class, y_pred)
print('F1 score: %f' % f1)

# OC-SVM

# svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
# print(svm)
# svm.fit(X_train)
# pred = svm.predict(X_test)
# print(pred)
# svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
# Improve the speed of svm

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

svm = OneClassSVM(kernel='rbf')
print(svm)
svm.fit(X_train[:12000])
pred = svm.predict(X_test)
print(pred)

print(pred)
pred = [1 if e == -1 else 0 for e in pred]
print(pred)
print(error_df.true_class)
precision = precision_score(error_df.true_class, pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(error_df.true_class, pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(error_df.true_class, pred)
print('F1 score: %f' % f1)
##### show oc-SVM figure ######
#import libraries
# https://medium.com/@mail.garima7/one-class-svm-oc-svm-9ade87da6b10
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.utils.example import visualize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

df = pd.read_csv("creditcard.csv")
data = df.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X_train, X_test = train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
y_train = X_train['Class']
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values

pca = PCA(n_components=2)

X_train = X_train[:12000]
# https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/
# reduce the dimesion to 2
pca.fit(X_train)
X_train = pca.transform(X_train)
y_train = y_train[:12000]
X_test = X_test[:10000]
pca.fit(X_test)
X_test = pca.transform(X_test)
y_test = y_test[:10000]
# scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_test = scaling.transform(X_test)

svm = OneClassSVM(nu=0.9, gamma=0.5, kernel='rbf')
print(svm)
svm.fit(X_train)
# pred = svm.predict(X_test)
# print(pred)
# binary labels
y_train_pred = svm.predict(X_train)
y_train_pred = [0 if e == -1 else 1 for e in y_train_pred]
y_test_pred = svm.predict(X_test)
y_test_pred = [0 if e == -1 else 1 for e in y_test_pred]
# prediction visualization
visualize(svm,
    X_train,
    y_train,
    X_test,
    y_test,
    y_train_pred,
    y_test_pred,
    show_figure=True,
    save_figure=False,)

