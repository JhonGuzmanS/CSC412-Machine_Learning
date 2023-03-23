import pandas as pd
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

index = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "y"]
features_index = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
df = pd.read_csv("car.txt", names=index)

df.replace("5more", '6', inplace=True)
df.replace("more", '6', inplace=True)
df.replace("low", '1', inplace=True)
df.replace("med", '2', inplace=True)
df.replace("high", '3', inplace=True)
df.replace("vhigh", '4', inplace=True)
df.replace("small", '1', inplace=True)
df.replace("big", '3', inplace=True)
df.replace("low", '1', inplace=True)


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # splits into other nodes
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calc_leaf(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_features):
        best_split = {}
        max_info_gain = -float(1)

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        return gain

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for label in class_labels:
            p = len(y[y == label]) / len(y)
            gini += p ** 2
        return 1 - gini

    def calc_leaf(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values.reshape(-1, 1)
average_score = []
size = [0.2, 0.4, 0.6, 0.8]

for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=rd.choice(size), random_state=41)

    classifier = DecisionTreeClassifier(min_samples_split=5, max_depth=4)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)
    average_score.append(score)

print("the average is: ", np.mean(average_score))
average_score.sort()
plt.plot([1,2,3,4,5,6,7,8,9,10], average_score)
plt.show()
