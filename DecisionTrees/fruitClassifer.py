
# we will also need scipy
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

"""
Here is our data in code:

features = [[140,"smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
labels = ["apple", "apple", "orange", "orange"]

we will use in our features: 0 == "bumpy" and 1 == "smooth";

same for the labels: 0 == "apple", 1 == "orange";

"""

# features are the measurements in grams and it texture is either
# 1 == bumpy or 0 == smooth
features = [[140,1], [130,1], [150, 0], [170, 1]]

# The labels are 0 and 1 where 0 ==  orange , 1 == apple
labels = [0, 0, 1, 1]

# creating an empty box of rules for apples and oranges
clf = tree.DecisionTreeClassifier()  # the decision tree classifier

# here is where it gets train
# fit is the algorithm to find patterns in the data

clf = clf.fit(features, labels)

# print the result according to its features
print(clf.predict([[170,1]]))




# output

# [1]  it's an orange!

"""
credits to google developers

https://youtu.be/cKxRvEZd3Mw

I just merely re-created it for my practice and for building on the
concepts.

Thank you

""""

# ==== end!
