import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
import pickle

trX, trY = np.load('features.npy'), np.load('labels.npy')

sns.set(style='ticks', palette='Set2')

def get_colors(n):
    return dict(zip(range(n), sns.color_palette('Set2', n)))

def calc_accuracy(trainX, trainY, C=0, n_estimators=0, max_depth=0, gamma=0, 
                        purity=None, alpha=0, solver=None, max_iter=0, coef0=0, model=None):
    
    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=purity)
    elif model == 'svm':
        clf = SVC(gamma=gamma, C=C, coef0=coef0) 
    elif model == 'mlp':
        clf = MLPClassifier(alpha=alpha, solver=solver, max_iter=max_iter)

    nfolds = 3
    N = len(trainY)
    fold = np.concatenate(np.random.randint(nfolds, size=(N,1)))

    fold_accuracies = []

    for f in range(nfolds):
        fold_count = 0
        trX = trainX[fold != f,:]
        trY = trainY[fold != f]
        tstX = trainX[fold == f,:]
        tstY = trainY[fold == f]

        clf.fit(trX, trY.ravel())

        for i in range(len(tstY)):
            y_hat = clf.predict(tstX[i].reshape(1, -1))
            if y_hat == tstY[i]:
                fold_count += 1
    fold_accuracy = fold_count / float(len(tstY))
    fold_accuracies.append(fold_accuracy)

    tst_acc = np.mean(fold_accuracies)

    train_count = 0
    clf.fit(trainX, trainY.ravel())
    for j in range(len(trainY)):
        y_hat = clf.predict(trainX[j].reshape(1, -1))

        if y_hat == trainY[j]:
            train_count += 1
    tr_acc = train_count / float(len(trainY))

    return tr_acc, tst_acc

"""
Let's try a random forest.
"""
print 'Random Forest Classifier'
plt.figure(1)
rf_out = open('rf.csv', 'w')
rf_out.write("tree_depth,purity_measure,num_estimators,test_accuracy\n")
ests = [1, 3, 5, 10]
depths = [1, 3, 5, 10]
purities = ['gini', 'entropy']
cs = get_colors(len(depths) * len(purities))
counter = 0
for d in depths:
    for purity in purities:
        fs_accs = []
        for e in ests:
            accuracies = calc_accuracy(trX, trY, n_estimators=e, max_depth=d, purity=purity, model='rf')
            print d, purity, e, accuracies
            rf_out.write("%s,%s,%s,%s,%s\n" % (d, purity, e, accuracies[0], accuracies[1]))
            fs_accs.append(accuracies)
        tr_acc = [item[0] for item in fs_accs]
        tst_acc = [item[1] for item in fs_accs]
        tr_label = "Random Forest, tree depth=%s, purity=%s -- training" % (d, purity)
        tst_label = "Random Forest, tree depth=%s, purity=%s -- testing" % (d, purity)
        plt.plot(ests, tr_acc, label=tr_label, color=cs[counter], ls='dashed')
        plt.plot(ests, tst_acc, label=tst_label, color=cs[counter])
        counter += 1
sns.despine(right=True, top=True)
plt.ylim([0,1.05])
plt.legend(loc=4, fontsize=8)
plt.ylabel('Accuracy')
plt.xlabel('Forest Size')
plt.savefig('rf.png', dpi=300)

"""
Let's try a support vector machine.
"""
print 'Support Vector Machine Classifier'
plt.figure(2)
svm_out = open('svm.csv', 'w')
svm_out.write("gamma,coef0,C,training_accuracy,test_accuracy\n")
gammas = [0.05, 0.1, 0.5] 
Cs = [0.1, 0.5, 1, 3]
coef0s = [0.05, 0.1, 0.5]
cs = get_colors(len(gammas) * len(coef0s))
counter = 0
for gamma in gammas:
    for coef0 in coef0s:
        svm_accs = []
        for C in Cs:
            accuracies = calc_accuracy(trX, trY, gamma=gamma, coef0=coef0, C=C, model='svm')
            print gamma, coef0, C, accuracies
            svm_out.write("%s,%s,%s,%s,%s\n" % (gamma, coef0, C, accuracies[0], accuracies[1]))
            svm_accs.append(accuracies)
        tr_acc = [item[0] for item in svm_accs]
        tst_acc = [item[1] for item in svm_accs]
        tr_label = "SVM, gamma=%s, coef0=%s -- training" % (gamma, coef0)
        tst_label = "SVM, gamma=%s, coef0=%s -- testing" % (gamma, coef0)
        plt.plot(Cs, tr_acc, label=tr_label, color=cs[counter], ls='dashed')
        plt.plot(Cs, tst_acc, label=tst_label, color=cs[counter])
        counter += 1
sns.despine(right=True, top=True)
plt.ylim([0,1.05])
plt.legend(loc=4, fontsize=8)
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.savefig('svm.png', dpi=300)

"""
Let's try a Multi-Layer Perceptron.
"""
print 'Multi-layer Perceptron'
plt.figure(3)
mlp_out = open('mlp.csv', 'w')
mlp_out.write("max_iter,solver,alpha,training_accuracy,test_accuracy\n")
max_iters = [50, 100, 500, 1000]
solvers = ['lbfgs', 'adam']
alphas = [0.01, 0.05, 0.1, 0.5, 1]
cs = get_colors(len(solvers) * len(max_iters))
counter = 0
for max_iter in max_iters:
    for solver in solvers:
        mlp_accs = []
        for alpha in alphas:
            accuracies = calc_accuracy(trX, trY, solver=solver, max_iter=max_iter, alpha=alpha, model='mlp')
            print max_iter, solver, alpha, accuracies
            mlp_out.write("%s,%s,%s,%s,%s\n" % (max_iter, solver, alpha, accuracies[0], accuracies[1]))
            mlp_accs.append(accuracies)
        tr_acc = [item[0] for item in mlp_accs]
        tst_acc = [item[1] for item in mlp_accs]
        tr_label = "MLP, max_iter=%s, solver=%s -- training" % (max_iter, solver)
        tst_label = "MLP, max_iter=%s, solver=%s -- testing" % (max_iter, solver)
        plt.plot(alphas, tr_acc, label=tr_label, color=cs[counter], ls='dashed')
        plt.plot(alphas, tst_acc, label=tst_label, color=cs[counter])
        counter += 1
sns.despine(right=True, top=True)
plt.ylim([0,1.05])
plt.legend(loc=4, fontsize=8)
plt.ylabel('Accuracy')
plt.xlabel('Alpha')
plt.savefig('mlp.png', dpi=300)



clf = MLPClassifier(alpha=0.01, solver='lbfgs', max_iters=50)
clf.fit(trX)


for i in trY:
    clf.predict(trX[counter])

