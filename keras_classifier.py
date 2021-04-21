from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras

from sklearn import metrics
import numpy as np



def classifier(input_shape=(128,), output_shape=2, hidden_size=256, if_print=0):

	model = Sequential()
	model.add(Dense(hidden_size, input_shape=input_shape, activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(output_shape, activation='softmax'))
	
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
	
	if if_print == 1:
		print(model.summary())

	return model

def fit_model(model, train_data, train_labels, val_data, val_labels, batch_size=16384, epochs=2000, verbose=0, validation_freq=1):
	
	callback = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10)
	model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback])
	
	return model

def predict(model, test_data, test_labels):

	preds = model.predict(test_data)

	pred_ts = np.argmax(preds, axis=1)

	# print(metrics.classification_report(test_labels, pred_ts))

	acc_score = metrics.accuracy_score(test_labels, pred_ts)

	recall_score = metrics.recall_score(test_labels, pred_ts, average='macro')

	f1_score = metrics.f1_score(test_labels, pred_ts, average='macro')

	map_score = metrics.average_precision_score(test_labels, pred_ts)

	roc_score = metrics.roc_auc_score(test_labels, pred_ts)

	fpr, tpr, thresholds = metrics.precision_recall_curve(test_labels, pred_ts)
	aupr_score = metrics.auc(tpr, fpr)

	scores = [acc_score, recall_score, f1_score, map_score, roc_score, aupr_score]

	return scores

def evaluate(train_data, train_labels, val_data, val_labels, test_data, test_labels):

	model = classifier()
	model = fit_model(model, train_data, train_labels, val_data, val_labels)
	scores = predict(model, test_data, test_labels)

	del model
	keras.backend.clear_session()

	return scores
