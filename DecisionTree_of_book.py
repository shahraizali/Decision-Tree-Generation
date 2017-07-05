from sklearn.datasets import load_iris 
import numpy as np
from sklearn import tree
#for age
    # 1 for young
    # 2 for pre-presbyopic
    # 3 fpr pres byopic

#for specRX
    #1 for myopia
    #2 for high hypermetropia 

#for astig
    #1 for no
    #2 for yes

#for tears 
    # 1 for reduced
    # 2 fpr normal
feature_names = [ 'age' , 'specRx' , 'astig' , 'tears' ]
    
features = [
            #age , specRx , astig , tears
            [1    ,   1    ,  1   ,   1   ] ,
            [1    ,   1    ,  1   ,   2   ] ,
            [1    ,   1    ,  2   ,   1   ] ,
            [1    ,   1    ,  2   ,   2   ] ,
            [1    ,   2    ,  1   ,   1   ] ,
            [1    ,   2    ,  1   ,   2   ] ,
            [1    ,   2    ,  2   ,   1   ] ,
            [1    ,   2    ,  2   ,   2   ] ,
            [2    ,   1    ,  1   ,   1   ] ,
            [2    ,   1    ,  1   ,   2   ] ,
            [2    ,   1    ,  2   ,   1   ] ,
            [2    ,   1    ,  2   ,   2   ] ,
            [2    ,   2    ,  1   ,   1   ] ,
            [2    ,   2    ,  1   ,   2   ] ,
            [2    ,   2    ,  2   ,   1   ] ,
            [2    ,   2    ,  2   ,   2   ] ,
            [3    ,   1    ,  1   ,   1   ] ,
            [3    ,   1    ,  1   ,   2   ] ,
            [3    ,   1    ,  2   ,   1   ] ,
            [3    ,   1    ,  2   ,   2   ] ,
            [3    ,   2    ,  1   ,   1   ] ,
            [3    ,   2    ,  1   ,   2   ] ,
            [3    ,   2    ,  2   ,   1   ] ,
            [3    ,   2    ,  2   ,   2   ] 
            ]
# for classes
    # 1 for hard contact lense
    # 2 for soft contact lense
    # 3 for no contact lense
labels = [3 , 2 , 3 , 1, 3, 2, 3, 1, 3, 2, 3, 1, 3 , 2 , 3 , 3 , 3 , 3 , 3 , 1 , 3 , 2 , 3 , 3]

# creating decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit( features , labels)

#print (test_target)
#print (test_data)
#print (clf.predict(test_data))

class_names = [ 'hard contact lense' , 'soft contact lense' , 'no centact lense' ]
# copied code
from sklearn.externals.six import StringIO
import pydotplus
dot_data =  StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names = feature_names,
                        class_names = class_names,
                        filled = True, rounded=True,
                        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("class.pdf")
graph.write_png("class.png")

# End of fild
