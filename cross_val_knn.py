num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(x_train, num_folds) #list of numpy arrays
y_train_folds = np.array_split(y_train, num_folds) #list of numpy arrays


acc = []
for j in k_choices:
	model_accuracy = list(range(num_folds))
	for i in range(num_folds):
		x_val = X_train_folds[i]
		y_val = y_train_folds[i]
		x_tr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
		y_tr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
		model = KNearestNeighbor()
		model.train(x_tr, y_tr)
		y_val_pred = model.predict(x_val, k=j, num_loops=0)
		model_accuracy[i] = (np.sum(y_val_pred == y_val))/len(y_val)
	acc.append(np.sum(model_accuracy)/num_folds)
	print(j, acc[-1])
