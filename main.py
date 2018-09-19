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


mask = list(range(5000))
x_train = x_train[:5000]
y_train = op.itemgetter(*mask)(y_train)

mask = list(range(500))
x_test = x_test[:500]
y_test = op.itemgetter(*mask)(y_test)

model = KNearestNeighbor()
model.train(x_train, y_train)

y_test_pred = model.predict(x_test, k=1, num_loops=0)
num = np.sum(y_test_pred == y_test)
print(num/500)
