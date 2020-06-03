import numpy as np
from models.node import Node


class DecisionTree:

    def __init__(self, x, y, max_depth=None, min_samples_leaf=None, sample_ixs=None, n_features=None,
                 attr_values=None, tree_type='C4.5'):
        self.tree = None
        self.feat_selected = np.zeros((x.shape[1], 1))

        if max_depth is None:
            self.max_depth = x.shape[1]
        else:
            self.max_depth = min(max_depth, x.shape[1])

        if min_samples_leaf is None:
            self.min_samples_leaf = 1
        else:
            self.min_samples_leaf = min_samples_leaf

        if sample_ixs is None:
            self.sample_ixs = list(range(x.shape[0]))
        else:
            self.sample_ixs = sample_ixs

        if n_features is None:
            self.n_features = x.shape[1]
        else:
            self.n_features = n_features

        if attr_values is not None:
            self.attr_vals = attr_values
        else:
            self.attr_vals = {}
            for i in range(x.shape[1]):
                self.attr_vals[i] = np.unique(x[:, i])

        if tree_type == 'C4.5':
            self.compute_score = self.compute_gain_ratio
        elif tree_type == 'ID3':
            self.compute_score = self.compute_info_gain
        elif tree_type == 'CART':
            self.compute_score = self.gini_index

        self.make_tree(x, y)

    def make_tree(self, x, y):
        self.tree = self.find_best_partition(x, y, list(range(x.shape[1])), 0)
        self.expand_tree(self.tree, x, y, 1)

    def find_best_partition(self, x, y, avail_attrs, depth):
        # Randomly select F features from the available ones
        np.random.shuffle(avail_attrs)
        feat_ixs = avail_attrs[:self.n_features]

        # Compute the score for each attribute and keep the one with the highest score
        best_score = -100
        for feat_ix in feat_ixs:
            score = self.compute_score(x[:, feat_ix], y)

            if score > best_score:
                best_feat_ix = feat_ix
                best_score = score

        # Annotate this feature as selected in the tree creation (To measure the feature importance in the forest)
        self.feat_selected[best_feat_ix] = 1

        # Remove the attribute from the list of available attributes
        avail_attrs = [attr for attr in avail_attrs if attr != best_feat_ix]

        # Create the Node and add a child per value of the selected attribute
        out_node = Node(attribute=best_feat_ix, avail_attrs=avail_attrs, depth=depth, children={})
        for val in self.attr_vals[best_feat_ix]:
            out_node.add_child(val, np.argwhere(x[:, best_feat_ix] == val)[:, 0])

        return out_node

    def expand_tree(self, tree, x, y, depth):
        considered_insts = tree.get_instances()

        for key, val in tree.children.items():
            if len(set(y[val])) == 1:
                # If there is only one class in this subset, set the child terminal value to this class
                tree.children[key] = Node(terminal_value=y[val[0]], depth=depth)
            elif len(val) == 0:
                # If the split left this branch empty, set the terminal value to the majority class of the parent subset
                labels, counts = np.unique(y[considered_insts], return_counts=True)
                terminal_value = labels[np.argmax(counts)]
                tree.children[key] = Node(terminal_value=terminal_value, depth=depth)
            elif self.min_samples_leaf > len(val) or depth == self.max_depth:
                # If the number of samples at this leaf is lower than the minimum or maximum depth has been reached,
                # set the terminal value to the majority class in the branch subset
                labels, counts = np.unique(y[val], return_counts=True)
                terminal_value = labels[np.argmax(counts)]
                tree.children[key] = Node(terminal_value=terminal_value, depth=depth)
            else:
                # Otherwise, find the best partition for this leaf and expand the subtree
                tree.children[key] = self.find_best_partition(x[val, :], y[val], tree.avail_attrs, depth)
                self.expand_tree(tree.children[key], x[val], y[val], depth+1)

        return

    def predict(self, x_tst):
        return self.check_node(x_tst, self.tree)

    def check_node(self, x_tst, tree):
        if tree.terminal_value is not None:
            return tree.terminal_value
        else:
            return self.check_node(x_tst, tree.children[x_tst[tree.attribute]])

    @staticmethod
    def compute_info_gain(feat_data, y):
        unique_labels, counts_labels = np.unique(y, return_counts=True)

        entropy_x = -sum([c/len(y) * np.log2(c/len(y)) for c in counts_labels])

        unique_vals, counts_vals = np.unique(feat_data, return_counts=True)

        info_x_attr = 0
        for val_ix in range(len(unique_vals)):
            sub_y = y[feat_data == unique_vals[val_ix]]
            unique_labels, counts_labels = np.unique(sub_y, return_counts=True)
            info_x_attr += counts_vals[val_ix]/len(y) * -sum([c/len(sub_y) * np.log2(c/len(sub_y)) for c in counts_labels])

        info_gain = entropy_x - info_x_attr

        return info_gain

    @staticmethod
    def compute_gain_ratio(feat_data, y):
        unique_labels, counts_labels = np.unique(y, return_counts=True)

        entropy_x = -sum([c / len(y) * np.log2(c / len(y)) for c in counts_labels])

        unique_vals, counts_vals = np.unique(feat_data, return_counts=True)

        info_x_attr = 0
        for val_ix in range(len(unique_vals)):
            sub_y = y[feat_data == unique_vals[val_ix]]
            unique_labels, counts_labels = np.unique(sub_y, return_counts=True)
            info_x_attr += counts_vals[val_ix] / len(y) * -sum(
                [c / len(sub_y) * np.log2(c / len(sub_y)) for c in counts_labels])

        info_gain = entropy_x - info_x_attr

        split_info = -sum([c/len(y) * np.log2(c/len(y)) for c in counts_vals])

        return info_gain / split_info if split_info != 0 else 0

    @staticmethod
    def gini_index(feat_data, y):
        score = 0

        unique_vals = np.unique(feat_data)

        for val in unique_vals:
            subset = np.argwhere(feat_data == val)[:, 0]
            labels, counts = np.unique(y[subset], return_counts=True)
            score += (1 - sum((counts / len(subset)) ** 2)) * len(subset) / (len(y))

        # The negative score is returned because the partition score needs to be maximized
        return -score

    def print_tree(self, attr_names):
        self.print_tree_aux('Root', self.tree, attr_names)

    def print_tree_aux(self, branch, tree, attr_names):
        if tree.terminal_value is None:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6 attr_\033[1m' +
                  str(attr_names[tree.attribute]) + '\033[0m')
            for branch, child_tree in tree.children.items():
                self.print_tree_aux(branch, child_tree, attr_names)
        else:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6 class_\033[94m' +
                  str(tree.terminal_value) + '\033[0m')
