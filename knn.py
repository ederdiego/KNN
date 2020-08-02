print(__doc__)

import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import itertools

def plot_classification_report(classificationReport,
                               title,
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:-4]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} (Supp:{1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()



k = int(sys.argv[1])
metrica = str(sys.argv[2])

# loads data
print ("Loading data...")
X, y = load_svmlight_file("features.txt")

# import some data to play with
class_names = [0,1,2,3,4,5,6,7,8,9]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 5)
X_train = X_train.toarray()
X_test = X_test.toarray()

# cria um kNN
neigh = KNeighborsClassifier(n_neighbors=k, metric=metrica)
#neigh = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

print ('Fitting knn')
neigh.fit(X_train, y_train)

# predicao do classificador
print ('Predicting...')
y_pred = neigh.predict(X_test)

# mostra o resultado do classificador na base de teste
print ('Accuracy: ',  neigh.score(X_test, y_test))
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
cm = confusion_matrix(y_test, y_pred)
print (cm)          
cr = classification_report(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])
print(cr)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("(K = "+str(k)+") (Metric = "+str(metrica)+") UnNormalized", None),
                  ("(K = "+str(k)+") (Metric = "+str(metrica)+") Normalized", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(neigh, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    #print(title)
    #print(disp.confusion_matrix)

plt.show()
plt.close()


plot_classification_report(cr, 'Classification Report (K = '+str(k)+') (Metric = '+metrica+')')
plt.show()
plt.close()
