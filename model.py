"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np 

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    values, counts = np.unique(labels , return_counts = True)
    p =  counts / counts.sum()
    G = 1 - np.sum(p**2)
    return G

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    mask   = features[:, feature_index] <= threshold  # ['True','False', ...]
    left_features = features[mask]
    left_labels = labels[mask]
    right_features = features[~mask]
    right_labels = labels[~mask]
    return (left_features , left_labels,right_features, right_labels)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n_left = len(left_labels)
    n_right = len(right_labels)
    n = len(parent_labels)
    weight_left  = n_left / n
    weight_right = n_right / n
    after = weight_left * impurity(left_labels) + weight_right * impurity(right_labels)
    return impurity(parent_labels) - after

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    best = {'feature_index': None, 'threshold': None, 'score': 0.0}

    for fi in feature_indices:  
        U = np.unique(features[:, fi]) # unique values 
        thresholds = (U[:-1] + U[1:]) / 2 # midpoints 
        for t in thresholds :
            lf,ll,rf,rl = split_dataset(features, labels, fi, t)
            if len(ll) == 0 or len(rl) == 0: continue
            S = split_score(labels, ll, rl)
            if S > best["score"]:
                best['feature_index'] = fi
                best['threshold'] = t
                best['score'] = S

    return best

# Step 5 - should_stop
import numpy as np 

def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    unique = np.unique(labels)
    if len(unique) == 1:
        return True
    elif depth >= max_depth :
        return True
    elif len(labels) < min_samples_split :
        return True
    else :
        return False

# Step 6 - leaf_prediction
import numpy as np 


def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    labels = np.array(labels)
    values, counts = np.unique(labels, return_counts = True)
    x = np.argmax(counts) # index of the large number
    return int(values[x])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    if feature_subset is None:
        candidates = range(features.shape[1])
    else:
        candidates = list(feature_subset)

    best = best_split(features, labels, candidates)
    if best['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    lf, ll, rf, rl = split_dataset(features, labels, best['feature_index'], best['threshold'])
    return  {'leaf': False, 'feature_index': best['feature_index'], 'threshold': best['threshold'], 'left': build_tree(lf, ll, max_depth, min_samples_split, feature_subset, depth + 1), 'right': build_tree(rf, rl, max_depth, min_samples_split, feature_subset, depth + 1)}

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if tree['leaf'] == True :
        return tree['prediction']
    else :
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if example[feature_index]<= threshold:
            return predict_example_tree(tree['left'], example)
        else :
            return predict_example_tree(tree['right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    preds = []
    for row in features :
        preds.append(predict_example_tree(tree, row))
    return np.array(preds, dtype= int)

# Step 10 - bootstrap_sample
import numpy as np 

def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n = len(features)
    idx = rng.integers(0, n, size=n)
    return features[idx], labels[idx]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

