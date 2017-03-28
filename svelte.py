from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve
from sklearn import svm
from sklearn import neighbors
import numpy as np
import pickle

trX, trY = np.load('1KG.features.npy'), np.load('1KG.labels.npy')
tstX, tstY = np.load('test.features.npy'), np.load('test.labels.npy')

def accuracy(trainX, trainY, testX, testY, C=0, n_estimators=0, max_depth=0, n_neighbors=0, degree=0, gamma=0, coef0=0, model=None):
    
    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(trainX, trainY.ravel())
    elif model == 'lr':
        clf = LogisticRegression(C=C)
        clf.fit(trainX, trainY.ravel())
    elif model == 'knn':
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(trainX, trainY.ravel())
    elif model == 'svm':
        clf = svm.SVC(C=0.5, kernel='poly', degree=5, gamma=1, coef0=0.05)
        clf.fit(trainX, trainY.ravel())


    #nfolds = 10
    #N = len(testY)
    #fold = np.concatenate(np.random.randint(nfolds, size=(N,1)))

    #fold_accuracies = []
    count = 0 
    #for f in range(nfolds):
    #    fold_count = 0
    #    trX = testX[fold != f,:]
    #    trY = testY[fold != f]
    #    tstX = testX[fold == f,:]
    #    tstY = testY[fold == f]

    #    clf.fit(trX, trY.ravel())

    for i in range(len(testY)):
        y_hat = clf.predict(testX[i].reshape(1, -1))
        if y_hat == tstY[i]:
            count += 1
    accuracy = count / float(len(testY))
    #fold_accuracies.append(fold_accuracy)


    tst_acc = accuracy

    train_count = 0
    clf.fit(trainX, trainY.ravel())
    for j in range(len(trainY)):
        y_hat = clf.predict(trainX[j].reshape(1, -1))

        if y_hat == trainY[j]:
            train_count += 1
    tr_acc = train_count / float(len(trainY))

    return tr_acc, tst_acc
"""
n_neighbors = [1,3,5,10, 20, 50]
for n in n_neighbors:
    print accuracy(trX, trY, tstX, tstY, n_neighbors=n, model='knn')

Cs = [0.001, 0.05, 0.1, 0.5, 1, 3, 5, 10, 20, 50, 100]
for C in Cs:
    print accuracy(trX, trY, tstX, tstY, C=C, model='lr')
"""
print 'support vector machine'
print accuracy(trX, trY, tstX, tstY, model='svm')

print 'logistic regression'
Cs = [0.001, 0.05, 0.1, 0.5, 1, 3, 5, 10, 20, 50, 100]
for C in Cs:
    print accuracy(trX, trY, tstX, tstY, model='lr', C=C)
