import numpy as np 
import pickle
def unpickle(file):
	with open(file, 'rb') as fo:
		d = pickle.load(fo, encoding='bytes')
	return d 

path = 'H:/learning-area/assignment1/cifar-10-batches-py/'
files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
label_list = []
data_array = np.zeros(1, 3072)
for i in range(5):
	temp = unpickle(path + files[i])
	data_array = np.concatenate((data_arr, temp[b'data']), axis=0)
	label_list += temp[b'labels']
print(np.shape(data_array))
print(len(label_list))
train_data = data_array[1:]

