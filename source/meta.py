from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np

def main():

	print("loading training set...")
	data0 = pickle.load(open("TrainFV_AlexNet_mult", "rb"))
	data1 = pickle.load(open("TrainFV_VGG_mult", "rb"))
	data2 = pickle.load(open("TrainFV_Inception_mult", "rb"))
	
	trainFV = []
	trainLabels = []

	for ind in range(len(data0)):
		fv = data0[ind][0].tolist() + data1[ind][0].tolist() + data2[ind][0].tolist()
		trainFV.append(fv)
		trainLabels.append(data0[ind][1])		


	parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10], 'decision_function_shape':['ovo', 'ovr']}
	svc = svm.SVC()
	clf = GridSearchCV(svc, parameters)
	clf.fit(trainFV, trainLabels)

	print("loading validation set...")
	data0_val = pickle.load(open("ValFV_AlexNet_mult", "rb"))
	data1_val = pickle.load(open("ValFV_VGG_mult", "rb"))
	data2_val = pickle.load(open("ValFV_Inception_mult", "rb"))

	ValFV = []
	ValLabels = []

	pred_alexnet = []
	pred_vgg = []
	pred_inception = []

	for ind in range(len(data0_val)):
		pred_alexnet.append(np.argmax(data0_val[ind][0]))
		pred_vgg.append(np.argmax(data1_val[ind][0]))
		pred_inception.append(np.argmax(data2_val[ind][0]))

		fv = data0_val[ind][0].tolist() + data1_val[ind][0].tolist() + data2_val[ind][0].tolist()
		ValFV.append(fv)
		ValLabels.append(data0_val[ind][1])

	preds = clf.predict(ValFV)

	print ("Final validation AlexNet result: %.2f" % getReslts(pred_alexnet, ValLabels))
	print ("Final validation VGG result: %.2f" % getReslts(pred_vgg, ValLabels))
	print ("Final validation Inception result: %.2f" % getReslts(pred_inception, ValLabels))
	print ("Final validation Meta-classification result: %.2f" % getReslts(preds, ValLabels))

	print("loading test set...")
	data0_test = pickle.load(open("TestFV_AlexNet_Multi", "rb"))
	data1_test = pickle.load(open("TestFV_VGG_mult", "rb"))
	data2_test = pickle.load(open("TestFV_Inception_Multi", "rb"))

	TestFV = []
	TestLabels = []

	pred_alexnet = []
	pred_vgg = []
	pred_inception = []

	for ind in range(len(data0_val)):
		pred_alexnet.append(np.argmax(data0_test[ind][0]))
		pred_vgg.append(np.argmax(data1_test[ind][0]))
		pred_inception.append(np.argmax(data2_test[ind][0]))

		fv = data0_test[ind][0].tolist() + data1_test[ind][0].tolist() + data2_test[ind][0].tolist()
		TestFV.append(fv)
		TestLabels.append(data0_test[ind][1])

	preds = clf.predict(TestFV)

	print ("Final test AlexNet result: %.2f" % getReslts(pred_alexnet, TestLabels))
	print ("Final test VGG result: %.2f" % getReslts(pred_vgg, TestLabels))
	print ("Final test Inception result: %.2f" % getReslts(pred_inception, TestLabels))
	print ("Final test Meta-classification result: %.2f" % getReslts(preds, TestLabels))
    
def getReslts(preds, labels):
	matrix = confusion_matrix(labels, preds)
	#print(matrix)
	acc = 0.0
	for ind in range(4):
		acc += matrix[ind][ind]/sum(matrix[ind, :])

	normalized_acc = acc/4
	return normalized_acc

if __name__ == '__main__':
	main()