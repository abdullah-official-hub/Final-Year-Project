import numpy as np
from matplotlib import pyplot as plt
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D,MaxPooling1D
'''
raw_data=[]
f1 = open('sectionsCount.csv','rt')
#next(f1)
data = csv.reader(f1)
for x in data:
  raw_data.append(x)
f1.close()

raw_data_PCA=[]
f1 = open('asmPCA.csv','rt')
f1=open('asmOperators.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data_PCA.append(x)
f1.close()
'''
raw_data=[]
f1 = open('sizes.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data.append(x)
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
#raw_data_sizes = tf.keras.utils.normalize(np.array(raw_data_sizes).astype(np.float32),axis=0)
#raw_data_PCA = tf.keras.utils.normalize(np.array(raw_data_PCA).astype(np.float32),axis=0)

raw_data = np.column_stack((raw_data,raw_data_bytes))
#raw_data = np.column_stack((raw_data,raw_data_sizes))
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

np.random.shuffle(test_data)

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


#Expanding dimensions.
print("Expainding Diamonsions of the dataset for CNN")
train_data = np.expand_dims(train_data,axis=-1)
test_data = np.expand_dims(test_data,axis=-1)

'''#Adding 1-C coding for train and test labels.
print("Adding 1-C coding on labels of the dataset")
train_labels1 = np.random.choice([0],size=(train_labels.shape[0],9))
for i in range(0,train_labels.shape[0]):
  train_labels1[i,int(train_labels[i])]=1
test_labels1 = np.random.choice([0],size=(test_labels.shape[0],9))
for i in range(0,test_labels.shape[0]):
  test_labels1[i,int(test_labels[i])]=1

train_labels=train_labels1
test_labels=test_labels1'''


#Building and compiling the model.
print("Building Model.")

tf.keras.backend.clear_session()
model = Sequential()
#conv1d layer 1
model.add(Conv1D(filters=64,kernel_size=1,activation='relu',input_shape=(num_features-1,1)))
#Maxpool1d layer 1
model.add(MaxPooling1D(pool_size=2,strides=None))
#dropout layer to fight with overfitting.
model.add(Dropout(0.25))
#conv1d layer 2
model.add(Conv1D(filters=64,kernel_size=1,activation='relu'))
#Maxpool1d layer 2
model.add(MaxPooling1D(pool_size=1, strides=None))
#Flatten layer to make data's dimension easily fit to dense layer.
model.add(tf.keras.layers.Flatten())
#first dense layer
model.add(Dense(128,activation='relu'))
#secon dense layer
model.add(Dense(9,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop' ,metrics=['accuracy'])
#fitting dataset on the model.
print("Starting training.")
model.summary()
'''
model.fit(np.array(train_data).astype(np.float32),np.array(train_labels),epochs=20,verbose=1)
val_loss, val_acc = model.evaluate(np.array(train_data),np.array(train_labels))
print(val_loss, val_acc)
model.save('cnnmodel.model')
'''
max_val_accu=0
for x in range(1,200):
  print(x)
  model.fit(np.array(train_data).astype(np.float32),np.array(train_labels),epochs=1,verbose=0)
  val_loss, val_acc = model.evaluate(np.array(test_data),np.array(test_labels),verbose=0)
  if int(val_acc * 100) > max_val_accu:
    max_val_accu = val_acc * 100
    model.save('./models/weights_bytes.model')
    print(max_val_accu)
print(max_val_accu)
model.save("./models/CNN_FileSizes+ByteCounts_Accuracy_"+str(max_val_accu)+".model")

