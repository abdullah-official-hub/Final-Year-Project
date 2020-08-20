import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf


raw_data_PCA=[]
f1 = open('asmOperators.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data_PCA.append(x)
f1.close()


raw_data_sizes=[]
f1 = open('sizes_both.csv','rt')
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



raw_data_section=[]
f1 = open('sectionsCount.csv','rt')
data = csv.reader(f1)
for x in data:
  raw_data_section.append(x)
f1.close()


label=[]
f1 = open('label.csv','rt')
data = csv.reader(f1)
for x in data:
  label.append(int(x[0])-1)
f1.close()


#raw_data = tf.keras.utils.normalize(np.array(raw_data).astype(np.float32),axis=0)
raw_data_section = tf.keras.utils.normalize(np.array(raw_data_section).astype(np.float32),axis=0)
raw_data_bytes = tf.keras.utils.normalize(np.array(raw_data_bytes).astype(np.float32),axis=0)
raw_data_sizes = tf.keras.utils.normalize(np.array(raw_data_sizes).astype(np.float32),axis=0)
#raw_data_PCA = tf.keras.utils.normalize(np.array(raw_data_PCA).astype(np.float32),axis=0)
raw_data_asm = tf.keras.utils.normalize(np.array(raw_data_asm).astype(np.float32),axis=0)

raw_data = raw_data_bytes
raw_data = np.column_stack((raw_data,raw_data_section))
#raw_data = np.column_stack((raw_data,raw_data_bytes))
raw_data = np.column_stack((raw_data,raw_data_sizes))
raw_data = np.column_stack((raw_data,raw_data_asm))


print(np.array(raw_data).shape)
print(np.array(label).shape)

raw_data = np.column_stack((raw_data,np.array(label)))

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
percentElements =  int(totalElements * 0.5)

train_data = train_data + data_raw[0:percentElements+1]
test_data = test_data + data_raw[percentElements+1:totalElements]


np.random.shuffle(train_data)
np.random.shuffle(train_data)
np.random.shuffle(train_data)

train_data = np.array(train_data)
test_data = np.array(test_data)

"""
print(train_data)
print(test_data)
"""

train_label=[]
test_label=[]
clean_train=[]
clean_test=[]
print("New......: ")
trainDataRows, trainDataCols = np.array(train_data).shape
testDataRows, testDataCols = np.array(test_data).shape
print(trainDataRows, trainDataCols)
print(testDataRows, testDataCols)

for x in range(0,len(train_data)):
  train_label.append(train_data[x][trainDataCols - 1])
  clean_train.append(train_data[x][0:trainDataCols - 1])


for x in range(0,len(test_data)):
  test_label.append(test_data[x][trainDataCols - 1])
  clean_test.append(test_data[x][0:trainDataCols - 1])


#print(np.array(train_label))
#print(np.array(clean_train))
#print(np.array(test_label))
#print(np.array(clean_test))

from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier()
#train
clf = clf.fit(clean_train, train_label)

#prediction
ans = clf.predict(clean_test)

#accuracy
np.sum(ans == test_label)/(np.array(test_label).shape)[0]
