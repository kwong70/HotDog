from skimage.io import imread, imshow
from sklearn.utils import shuffle
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
import os

#import our data

#249 training examples
X_train = []
y_train = []
for picture_name in os.listdir('data/train/hot_dog'):
    image = imread('data/train/hot_dog/'+picture_name)
    image = resize(image, (300, 300))
    X_train.append(image)
    y_train.append(1)

for picture_name in os.listdir('data/train/not_hot_dog'):
    image = imread('data/train/not_hot_dog/'+picture_name)
    image = resize(image, (300, 300))
    X_train.append(image)
    y_train.append(0)


#250 testing examples
X_test = []
y_test = []
for picture_name in os.listdir('data/test/hot_dog'):
    image = imread('data/test/hot_dog/'+picture_name)
    image = resize(image, (300, 300))
    X_test.append(image)
    y_test.append(1)

for picture_name in os.listdir('data/test/not_hot_dog'):
    image = imread('data/test/not_hot_dog/'+picture_name)
    image = resize(image, (300, 300))
    X_test.append(image)
    y_test.append(0)


#scramble the test/training sets
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

image = X_train[0]

(image)

#model = MLPClassifier(alpha=1, max_iter=1000)

#model.fit(X_train, y_train)

#score = model.score(X_test, y_test)

#print(score)

