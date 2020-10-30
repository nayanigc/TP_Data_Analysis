import sklearn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# QUESTION 1 -
def load_Iris(Iris_path='Iris.xls'):
    csv_path= os.path.join(os.getcwd(), Iris_path)
    dataset= pd.read_excel(csv_path)
    return dataset

iris = load_Iris()
iris.head()
iris.info()
iris['sepal width'].value_counts()

#Question 2 -
print('la moyenne :' + str( iris.mean(axis=0)))
print(" l'ecart-type:"+ str(iris.std()))


#Question 3-

data = pd.DataFrame(columns=['iris_setosa','iris_versicolor','iris_virginica','S (random)'])
test = 0
while (test < 30):
    s = [10,30,50,70]
    tirage = np.random.choice(s)
    iris_random = iris.sample(n=tirage, random_state=1)
    iris_setosa = iris_random[iris_random['iris'] =='Iris-setosa']
    iris_setosa_moyenne=iris_setosa['sepal length'].mean(axis=0)
    iris_versicolor=iris_random[iris_random['iris'] =='Iris-versicolor']
    iris_versicolor_moyenne = iris_versicolor['sepal length'].mean(axis=0)
    iris_virginica = iris_random[iris_random['iris'] =='Iris-virginica']
    iris_virginica_moyenne = iris_virginica['sepal length'].mean(axis=0)
    data = data.append({'iris_setosa': iris_setosa_moyenne, 'iris_versicolor': iris_versicolor_moyenne, 'iris_virginica': iris_virginica_moyenne,'S (random)': tirage}, ignore_index=True)
    test = test+1
print(data)

#print('iris shape:', iris.shape)

# Question 4
print('----------------correlation(matrice)---------------------')
print(iris.corr())

# Question 5
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris,test_size=0.2,random_state =42)


#Question 6 
def int_labels_from_str_labels(str_labels, classes, binary_class=None):
    str_labels = str_labels.to_numpy()
    nb_labels = str_labels.shape[0]
    y_train = np.zeros((nb_labels))

    for i in range(nb_labels):
        if binary_class is not None:
            y_train[i] = 1 if str_labels[i] == binary_class else 0
        else:
            int_label = classes.index(str_labels[i])
            y_train[i] = int_label
    
    return y_train

classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']

y_train_str = train_set['iris']
y_train = int_labels_from_str_labels(y_train_str, classes)

print('----------------y-train-str-------------------------------------------')
print(y_train_str)
print('----------------y-train-------------------------------------------')
print(y_train)
print('-----------------X_train-----------------------------------------------')
X_train=train_set.drop(['iris'],axis=1)
print(X_train)

y_test_str = test_set['iris']
y_test = int_labels_from_str_labels(y_test_str, classes)
X_test=test_set.drop(['iris'],axis=1)
print('---------------------------------------')
#Question 7
from sklearn.metrics import mean_squared_error
from sklearn import tree
# max-depth : profondeur de l'arbre variant de 2 ou 3

# Max leaf node = 1 pas OK

for max_depth in [2, 3]:    
    dtree= tree.DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=2)
    dtree.fit(X_train, y_train)
    housing_predictions= dtree.predict(X_train)
    lin_mse = mean_squared_error(y_train, housing_predictions)
#Question 8    
    print('Erreur (train), max depth=', str(max_depth), ': ', lin_mse)
    housing_predictions= dtree.predict(X_test)
    lin_mse = mean_squared_error(y_test, housing_predictions)
    print('Erreur (test), max depth=', str(max_depth), ': ', lin_mse)
#Question 9:
# Le modele obtenue est un peu sur-apprentissage car le modele de train est de 
#0.325 alors le modele de test est qu'on a lui de 0.366.
    
# Question 10
from sklearn.model_selection import  cross_val_score    
scores= cross_val_score(dtree,X_train,y_train,scoring='neg_mean_squared_error',cv=3)
print('-------------------------------------------------------')
print('le score pour cv=3 :',scores)
print('-------------------------------------------------------')
#Qestion 12
from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier(loss='log')

y_train_str = train_set['iris']
y_train = int_labels_from_str_labels(y_train_str, classes,'Iris-virginica')

y_test_str = test_set['iris']
y_test = int_labels_from_str_labels(y_test_str, classes,'Iris-virginica')
print('---------------------------------------')
sgd_clf.fit(X_train,y_train)
print('------------------------------------------------------')
print('score', cross_val_score(sgd_clf,X_train,y_train,scoring='accuracy',cv=3))
print('-------------------------------------------------------')
#Question 13
y_train_str = train_set['iris']
y_train = int_labels_from_str_labels(y_train_str, classes)
print('----------------y-train-str-------------------------------------------')
print(y_train_str)
print('----------------y-train-------------------------------------------')
print(y_train)
print('-----------------X_train-----------------------------------------------')
X_train=train_set.drop(['sepal length','petal width', 'iris'],axis=1)
print(X_train)

y_test_str = test_set['iris']
y_test = int_labels_from_str_labels(y_test_str, classes)
X_test=test_set.drop(['sepal length','petal width', 'iris'],axis=1)
print('---------------------------------------')

from sklearn.metrics import mean_squared_error
from sklearn import tree
# max-depth : profondeur de l'arbre variant de 2 ou 3

# Max leaf node = 1 pas OK

for max_depth in [2, 3]:    
    dtree= tree.DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=2)
    dtree.fit(X_train, y_train)
    housing_predictions= dtree.predict(X_train)
    lin_mse = mean_squared_error(y_train, housing_predictions)
    print('Erreur (train), max depth=', str(max_depth), ': ', lin_mse)
    housing_predictions= dtree.predict(X_test)
    lin_mse = mean_squared_error(y_test, housing_predictions)
    print('Erreur (test), max depth=', str(max_depth), ': ', lin_mse)
plt.show()"""