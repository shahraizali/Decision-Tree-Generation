from sklearn.datasets import load_iris 
import numpy as np
from sklearn import tree
iris =  load_iris()
test_ids = [ 0 ,  50 , 100]
print ( iris.feature_names )
#print ( iris.target_names)
#print (iris.data[0])
#print (iris.target[0])
 
    
# creating training  data 
train_target = np.delete( iris.target, test_ids ) 
train_data =  np.delete( iris.data , test_ids , axis = 0)

# testing data
test_target =  iris.target[test_ids]
test_data =  iris.data[ test_ids ]

clf  = tree.DecisionTreeClassifier()
clf  = clf.fit(train_data , train_target)

#print (test_target)
#print (test_data)
#print (clf.predict(test_data))


# copied code
from sklearn.externals.six import StringIO
import pydotplus
dot_data =  StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names = iris.feature_names,
                        class_names = iris.target_names,
                        filled = True, rounded=True,
                        impurity=False)

pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png("iris_wala.png")