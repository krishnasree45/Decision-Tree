import pandas as pd
from sklearn import model_selection
import time
import math
import tracemalloc
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# metric = "info_gain"
# metric = "gini"
metric = "gain_ratio"

def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        target = row[-1]
        if target not in counts:
            counts[target] = 0
        counts[target] += 1
    return counts


def max_label(dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class DatasetPartitionPoint:
    """A DatasetPartitionPoint is used to partition a dataset.

    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value


def partition(rows, dataset_partition_point):
    
    true_rows, false_rows = [], []
    for row in rows:
        if dataset_partition_point.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def entropy(rows):

    # compute the entropy.
    entries = class_counts(rows)
    avg_entropy = 0
    size = float(len(rows))
    for label in entries:
        prob = entries[label] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1*avg_entropy


def info_gain(left, right, current_uncertainty):
    
    p = float(len(left)) / (len(left) + len(right))

    ## TODO: Step 3, Use Entropy in place of Gini
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def split_ratio(left, right):
    d1 = float(len(left)) / (len(left) + len(right))
    d2 = 1 - d1
    if d1== 0 or d2 ==0:
        return -1
    return -1*(d1*math.log(d1,2)+ d2*math.log(d2,2))

def gain_ratio(info_gain, split_ratio):
    return float(info_gain/split_ratio)

def find_best_split(rows, header, metric):
    
    best_gain = 0  # keep track of the best information gain
    best_dataset_partition_point = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns
    rows_len = len(rows)
    gini_total_dataset = gini(rows)

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            dataset_partition_point = DatasetPartitionPoint(col, val, header)

            left, right = partition(rows, dataset_partition_point)
            if metric == "gini":   
                gini_left = gini(left)
                gini_right = gini(right)
                gini_of_feature = ((len(left)/ rows_len) * gini_left) + ((len(right)/rows_len) * gini_right)
                gini_delta = gini_total_dataset - gini_of_feature
                if(gini_delta >= best_gain):
                    best_gain, best_dataset_partition_point = gini_delta, dataset_partition_point

            elif metric == "info_gain":
                gain = info_gain(left, right, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_dataset_partition_point = gain, dataset_partition_point

            else: # Gain Ratio
                gain_ratio_cal = gain_ratio(info_gain(left, right, current_uncertainty), split_ratio(left, right))
                if gain_ratio_cal >= best_gain:
                    best_gain, best_dataset_partition_point = gain_ratio_cal, dataset_partition_point

            if len(left) == 0 or len(right) == 0:
                continue

    return best_gain, best_dataset_partition_point

## TODO: Step 2
class Leaf:
    """A Leaf node classifies data.

    """

    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth

class Decision_Node:
    """A Decision Node stores a dataset_partition_point.

    This holds a reference to the dataset_partition_point, and to the two child nodes.
    """

    def __init__(self,
                 dataset_partition_point,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows):
        self.dataset_partition_point = dataset_partition_point
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows


def build_tree(rows, header, depth=0, id=0):
    

    gain, dataset_partition_point = find_best_split(rows, header, metric)
    # gain, DatasetPartitionPoint = find_best_split(rows, header, "gini")
    # gain, DatasetPartitionPoint = find_best_split(rows, header, "gain_ratio")


    if gain == 0:
        return Leaf(rows, id, depth)

    true_rows, false_rows = partition(rows, dataset_partition_point)

    true_branch = build_tree(true_rows, header, depth + 1, 2 * id + 2)

    false_branch = build_tree(false_rows, header, depth + 1, 2 * id + 1)

    return Decision_Node(dataset_partition_point, true_branch, false_branch, depth, id, rows)

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predicted_label

    if node.dataset_partition_point.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_tree(node, spacing=""):

    if isinstance(node, Leaf):
        print(spacing + "Leaf id: " + str(node.id) + " Predictions: " + str(node.predictions) + " Label Class: " + str(node.predicted_label))
        return

    print(spacing + str(node.dataset_partition_point) + " id: " + str(node.id) + " depth: " + str(node.depth))

    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def getLeafNodes(node, leafNodes =[]):

    if isinstance(node, Leaf):
        leafNodes.append(node)
        return

    getLeafNodes(node.true_branch, leafNodes)

    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes


def getInnerNodes(node, innerNodes =[]):

    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    getInnerNodes(node.true_branch, innerNodes)

    getInnerNodes(node.false_branch, innerNodes)

    return innerNodes

def computeAccuracy(rows, node, writeToFile = False):

    if writeToFile is True:
        file = open('final_data.csv', 'w')
        writer = csv.writer(file)
        list = ["Predicted", "Actual class"]
        writer.writerow(list)
    count = len(rows)
    if count == 0:
        return 0
    y_pred = []
    y_true = []
    accuracy = 0
    for row in rows:
        if writeToFile is True:
            list = [row[-1], classify(row, node)]
            writer.writerow(list)
        y_true.append(row[-1])
        y_pred.append(classify(row, node))
        if row[-1] == classify(row, node):
            accuracy += 1
    
    # conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # print('precision score: ',precision_score(y_true=y_true, y_pred=y_pred) )
    return round(accuracy/count, 2)

def preprocessing(rows):
    rows_modified = rows
    for row in rows:
        for i in row:
            if i == '' or i == '?':
                rows_modified.remove(row)
                break
    return rows_modified


# default data set
# df = pd.read_csv('data_set/car-evaluation-4classes.csv') # Working good
# df = pd.read_csv('data_set/contraceptive-method-choice.csv') # Working good
# df = pd.read_csv('data_set/arrhythmia-16classes.csv') # Working Good
# df = pd.read_csv('data_set/credit-approval-2classes.csv') # Working good
# df = pd.read_csv('data_set/poker-hand-training-true-9classes.csv') # Working good
# df = pd.read_csv('data_set/glass identification - 7 classes.csv') # Working good
# df = pd.read_csv('data_set/tic-tac-toe-2classes.csv') # Working good
df = pd.read_csv('data_set/breast-cancer.csv') # Working good
# df = pd.read_csv('data_set/data_banknote_authentication-2classes.csv') # Working good
# df = pd.read_csv('data_set/breast-cancer-wisconsin-2classes.csv') # Working Good
# df = pd.read_csv('data_set/lung-cancer.csv') # Working Good
# df = pd.read_csv('data_set/iris.csv') # Working Good
# df = pd.read_csv('data_set/soybean-large.csv') # Working Good
# df = pd.read_csv('data_set/tae.csv') # Working Good
# df = pd.read_csv('data_set/Maternal Health Risk Data Set.csv') # Working Good


header = list(df.columns)

lst = df.values.tolist()

trainDF, testDF = model_selection.train_test_split(lst, test_size=0.2)

start = time.time()
tracemalloc.start()

trainDF = preprocessing(trainDF)
t = build_tree(trainDF, header)
current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

end = time.time()


time_elapsed = end-start
print("Time elapsed: ", time_elapsed)
current = current/ (10 ** 6)
print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
# get leaf and inner nodes
print("\nLeaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\nNon-leaf nodes ****************")
innerNodes = getInnerNodes(t)

for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

# print tree
maxAccuracy = computeAccuracy(testDF, t, True)
with open('final_data-metrics.csv', 'a') as f:
    # list = ["Info_Gain", maxAccuracy, time_elapsed, current]
    list = [metric, maxAccuracy, time_elapsed, current]
    # list = ["Gain_ratio", maxAccuracy, time_elapsed, current]

    writer = csv.writer(f)
    writer.writerow(list)
print("\nTree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
# print_tree(t)
