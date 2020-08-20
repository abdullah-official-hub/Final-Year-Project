import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
import csv
from joblib import dump, load




# load the iris dataset
#iris = datasets.load_iris()

raw_data=[]
f1 = open('sectionsCount.csv','rt')
#next(f1)
data = csv.reader(f1)
for x in data:
  raw_data.append(x)
f1.close()
'''
raw_data=[]
f1=open('asmPCA.csv','rt')
#f1=open('asmOperators.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data.append(x)
f1.close()
'''
raw_data_sizes=[]
f1 = open('sizes.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data_sizes.append(x)
f1.close()

raw_data_bytes=[]
f1 = open('byteCount.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data_bytes.append(x)
f1.close()

label=[]
f1 = open('label.csv','rt')
data = csv.reader(f1)
for x in data:
  label.append(int(x[0])-1)
f1.close()
print("Data set Loaded\n")


raw_data = tf.keras.utils.normalize(np.array(raw_data).astype(np.float32),axis=0)
raw_data_bytes = tf.keras.utils.normalize(np.array(raw_data_bytes).astype(np.float32),axis=0)

raw_data_sizes = tf.keras.utils.normalize(np.array(raw_data_sizes).astype(np.float32),axis=0)
#raw_data_PCA = tf.keras.utils.normalize(np.array(raw_data_PCA).astype(np.float32),axis=0)

raw_data = np.column_stack((raw_data,raw_data_bytes))
raw_data = np.column_stack((raw_data,raw_data_sizes))
#raw_data = np.column_stack((raw_data,raw_data_PCA))
raw_data = np.column_stack((raw_data,np.array(label)))
del label[:]
print("NP Array Created\n")
flag=0
data_raw=[]
train_data=[]
test_data=[]
for x in raw_data:
  if int(x[len(x)-1]) != flag:
    totalElements = len(data_raw)
    percentElements =  int(totalElements * 0.7)
    train_data = train_data + data_raw[0:percentElements+1]
    test_data = test_data + data_raw[percentElements+1:totalElements]
    flag = flag + 1
    data_raw=[]
  else:
    data_raw.append(x)

totalElements = len(data_raw)
percentElements =  int(totalElements * 0.7)
#Emptying Numpy Array raw_data
raw_data = np.zeros(1)

train_data = train_data + data_raw[0:percentElements+1]
test_data = test_data + data_raw[percentElements+1:totalElements]
#Emptying Numpy Array data_raw
data_raw = np.zeros(1)

np.random.shuffle(train_data)
np.random.shuffle(train_data)
np.random.shuffle(train_data)


train_data = np.array(train_data)
num_features = train_data.shape
num_features=num_features[1]

train_labels=train_data[:,num_features-1]
train_data = train_data[:,:-1]
#train_data = np.delete(train_data,num_features-1,1)

test_data = np.array(test_data)
test_labels=test_data[:,num_features-1]
test_data = test_data[:,:-1]
#test_data=np.delete(test_data,num_features-1,1)
print("Test Train Split Completed\n")

model  = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(np.array(train_data).astype(np.float32),np.array(train_labels).T,epochs=20,verbose=1)
model.summary()
max_val_accu=0
for x in range(1,1000):
  print(x)
  model.fit(np.array(train_data).astype(np.float32),np.array(train_labels).T,epochs=1,verbose=0)
  val_loss, val_acc = model.evaluate(np.array(test_data),np.array(test_labels).T,verbose=0)
  if int(val_acc * 100) > max_val_accu:
    max_val_accu = val_acc * 100
    #model.save('NN_weights_sectioncounts+bytecount+filesize_acc.model')
    print(max_val_accu)

model.fit(np.array(train_data).astype(np.float32),np.array(train_labels).T,epochs=1)

val_loss, val_acc = model.evaluate(np.array(test_data),np.array(test_labels).T)
print(val_loss, val_acc)
model.save('NN_weights_sectioncounts+bytecounts+filesizes_'+str(max_val_accu))
print(max_val_accu)