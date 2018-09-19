import numpy as np 
import pickle
def unpickle(file):
	with open(file, 'rb') as fo:
		d = pickle.load(fo, encoding='bytes')
	return d 

path = 'H:/learning-area/assignment1/cifar-10-batches-py/'
files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
y_train = []
data_array = np.zeros(1, 3072)
for i in range(5):
	temp = unpickle(path + files[i])
	data_array = np.concatenate((data_array, temp[b'data']), axis=0)
	y_train += temp[b'labels']
x_train = data_array[1:]
print(np.shape(x_train))
print(len(y_train))

temp = unpickle(path + files[5])
x_test = temp[b'data']
y_test = temp[b'labels']
