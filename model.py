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

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

