from skimage.io import imread
from sklearn.utils import shuffle
from skimage.transform import resize
import os
import numpy as np
import threading
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#import our data

#249 training examples
X_train = np.ndarray((498, 90000))
y_train = []
count = 0
for picture_name in os.listdir('data/train/hot_dog'):
    image = imread('data/train/hot_dog/'+picture_name, as_gray=True)
    image = resize(image, (300, 300))
    image = image.reshape((1, -1))
    X_train[count] = image
    count += 1
    y_train.append(1)


for picture_name in os.listdir('data/train/not_hot_dog'):
    image = imread('data/train/not_hot_dog/'+picture_name, as_gray=True)
    image = resize(image, (300, 300))
    image = image.reshape((1, -1))
    X_train[count] = image
    count += 1
    y_train.append(0)

#250 testing examples
X_test = np.ndarray((500, 90000))
y_test = []
count = 0
for picture_name in os.listdir('data/test/hot_dog'):
    image = imread('data/test/hot_dog/'+picture_name, as_gray=True)
    image = resize(image, (300, 300))
    image = image.reshape((1, -1))
    X_test[count] = image
    count += 1
    y_test.append(1)


for picture_name in os.listdir('data/test/not_hot_dog'):
    image = imread('data/test/not_hot_dog/'+picture_name, as_gray=True)
    image = resize(image, (300, 300))
    image = image.reshape((1, -1))
    X_test[count] = image
    count += 1
    y_test.append(0)


#scramble the test/training sets
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)



classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

threads = []

def fit_and_score(name, model, X_train, y_train, X_test, y_test):
    print("Starting " + name)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(name + ": " + str(score))

for name, model in zip(names, classifiers):
    thread = threading.Thread(target = fit_and_score, args = (name, model, X_train, y_train, X_test, y_test,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()


