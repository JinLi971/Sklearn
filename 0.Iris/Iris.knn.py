import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#load dataset
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#print the kinds of flowers
#print np.unique(iris_y)
print "data size:", iris_X.shape, iris_y.shape
#print iris_y
# Note that flowers are in order
# permute data to generate trainning and testing dataset



#test for once
np.random.seed()
indices = np.random.permutation(len(iris_X))

number_testing = 20

#let the last number_tesing data to be testing dataset, and the rest to
#be the trainning dataset
iris_X_train = iris_X[indices[:-number_testing]]
iris_y_train = iris_y[indices[:-number_testing]]
iris_X_test  = iris_X[indices[-number_testing:]]
iris_y_test  = iris_y[indices[-number_testing:]]

print "training data size:", iris_X_train.shape, "test data size", iris_X_test.shape

#initialize knn classifier
knn = KNeighborsClassifier(n_neighbors=10, weights='distance',algorithm='auto', leaf_size=10, p=2)

#train the classifier
knn.fit(iris_X_train, iris_y_train)

pred = knn.predict(iris_X_test) 
#predict
print "prediction:", pred

#compare with the results
print "truth:", iris_y_test

if (pred-iris_y_test).sum() == 0:
    print "Correct"
else:
    print "Opps... Something went wrong in the prediction"


#find the best K

K_min = 3
K_max = 20
n_bootstrap = 100
print "find the best K in (", K_min, ",", K_max, ") using the mean of 100 runs for each K"
U = np.zeros(K_max - K_min)
T = np.zeros(n_bootstrap)
for n_neighbors in range(K_min,K_max):
    for i in range(n_bootstrap):
        np.random.seed()
        indices = np.random.permutation(len(iris_X))
        iris_X_train = iris_X[indices[:-number_testing]]
        iris_y_train = iris_y[indices[:-number_testing]]
        iris_X_test  = iris_X[indices[-number_testing:]]
        iris_y_test  = iris_y[indices[-number_testing:]]
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(iris_X_train, iris_y_train)
        pred = knn.predict(iris_X_test)
        diff = pred-iris_y_test
        #print diff
        T[i] = np.count_nonzero(diff)
    U[n_neighbors-K_min] = T.mean()

print "error: ", U, "min error: ", U.min(), "min error at K = ", np.argmin(U)+K_min
