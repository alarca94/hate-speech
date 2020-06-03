import numpy as np


class PRISM_Algorithm:
    CONDITIONS = 'conditions'
    LABEL = 'label'

    def __init__(self, attr_names, attr_values):
        self.attr_names = attr_names
        self.attr_values = attr_values
        self.rules = []
        self.coverages = []
        self.precisions = []

    def _print_rule(self, rule, rule_id, add_prec_cov):
        if rule is None:
            rule = self.rules[rule_id]

        conds = ['(' + self.attr_names[condition[0]] + ', ' +
                 str(self.attr_values[self.attr_names[condition[0]]][condition[1]]) + ')'
                 for condition in rule[self.CONDITIONS]]

        if add_prec_cov:
            print(' \u2229 '.join(conds) + ' \u27f6 ' + str(rule[self.LABEL]) +
                  '\n### Precision: ' + str(round(self.precisions[rule_id] * 100, 2)) +
                  '%; Coverage: ' + str(round(self.coverages[rule_id] * 100, 2)) + '%\n')
        else:
            print(' \u2229 '.join(conds) + ' \u27f6 ' + str(rule[self.LABEL]))

    def _print_rules(self):
        print('')
        print('\033[1mTHE SET OF RULES IS:\033[0m')
        print('--------------------------------------------')
        print('--------------------------------------------')

        for r in range(len(self.rules)):
            self._print_rule(None, r, True)

        print('--------------------------------------------')
        print('Total coverage is: ' + str(round(sum(self.coverages) * 100, 2)) + '%')
        print('--------------------------------------------')
        print('--------------------------------------------')

    def fit(self, trn_data, trn_targets, log=False):
        # Check that the training set contains instances of more than one classification
        distinct_trn_targets, count_trn_targets = np.unique(trn_targets, return_counts=True)

        if len(distinct_trn_targets) > 1:
            trn_instances = np.concatenate((trn_data, trn_targets.reshape(len(trn_targets),1)), axis=1).tolist()

            for i in range(len(distinct_trn_targets)):
                curr_class_instances = [inst for inst in trn_instances if inst[-1] == distinct_trn_targets[i]]

                # STEP 5: Repeat STEPS 1-4 until no instance of the current class remains
                while len(curr_class_instances) > 0:
                    covered_instances = trn_instances[:]

                    # STEP 3: Repeat STEP 1 and 2
                    rule = {self.CONDITIONS: [], self.LABEL: distinct_trn_targets[i]}

                    available_attrs = list(range(len(self.attr_names)))

                    # While the current rule covers rows belonging to more than one class
                    while len(set([inst[-1] for inst in covered_instances])) > 1 \
                            and len(available_attrs) > 0:
                        biggest_p = 0
                        best_positive = 0

                        # STEP 1: Calculate all probabilities of occurrence, p(class_n | attribute_v)
                        for j in available_attrs:
                            attr_name = self.attr_names[j]
                            poss_values = self.attr_values[attr_name]

                            # Filter the selectors left to classify the remaining class instances
                            poss_values_left = set([inst[j] for inst in curr_class_instances if inst in covered_instances])

                            for k in range(len(poss_values)):
                                if poss_values[k] in poss_values_left:
                                    # Attr-value frequency in remaining training set (not limited to this class)
                                    total_instances = [inst for inst in covered_instances
                                                       if inst[j] == poss_values[k]]

                                    # Num. of samples of current class with this attr-value
                                    positive_instances = [inst for inst in total_instances
                                                          if inst[-1] == distinct_trn_targets[i]]

                                    n_positive = len(positive_instances)
                                    n_total = len(total_instances)

                                    if n_positive == 0:
                                        p = 0
                                    else:
                                        p = n_positive / n_total

                                    # STEP 2: Obtain best partition (attribute-value)
                                    if p > biggest_p:
                                        biggest_p = p
                                        attr_idx = j
                                        val_idx = k
                                        best_positive = n_positive
                                    elif p == biggest_p:
                                        # In case of tie, select the most frequent term
                                        if n_positive > best_positive:
                                            biggest_p = p
                                            attr_idx = j
                                            val_idx = k
                                            best_positive = n_positive

                        # Add term to the rule
                        rule[self.CONDITIONS].append((attr_idx, val_idx))

                        # Subset comprising all instances with the selected attribute-value (maybe different class)
                        covered_instances = [inst for inst in covered_instances
                                             if inst[attr_idx] == self.attr_values[self.attr_names[attr_idx]][val_idx]]

                        available_attrs.remove(attr_idx)

                    # Add rule to the set of rules
                    self.rules.append(rule)

                    # Add coverage of the current rule
                    self.coverages.append(len(covered_instances)/len(trn_targets))
                    self.precisions.append(biggest_p)

                    # STEP 4: remove all instances covered by this rule
                    curr_class_instances = [inst for inst in curr_class_instances if inst not in covered_instances]

            order_of_coverage = np.argsort(self.coverages)[::-1]
            self.rules = np.array(self.rules)[order_of_coverage]
            self.coverages = np.array(self.coverages)[order_of_coverage]

            if log:
                self._print_rules()

    def predict(self, tst_data, groundtruth=None, log=False):
        predicted_labels = []
        for row in range(tst_data.shape[0]):
            if log:
                print('\033[94m' + str(row) + '.\033[0m Analyzing row: ' + str(tst_data[row, :]))

            satisfied_rules = []
            not_visited_rules = self.rules.tolist()

            # Vector to contain the satisfied conditions
            true_conds = []

            while len(not_visited_rules) > 0:
                # Always take the first of the filtered rules (It will satisfy all true_conditions)
                valid_rule = True
                for cond in not_visited_rules[0][self.CONDITIONS]:
                    # If current condition is not satisfied yet...
                    if cond not in true_conds:
                        # ... and its value does not correspond to the sample attribute-value, filter rules
                        if self.attr_values[self.attr_names[cond[0]]][cond[1]] != tst_data[row, cond[0]]:
                            not_visited_rules.pop(0)

                            purge_rule_ids = self._filter_rules(not_visited_rules, cond, is_right_cond=False)
                            for idx in purge_rule_ids:
                                not_visited_rules.pop(idx)

                            valid_rule = False
                            break
                        else:
                            # Add new conditions to already satisfied conditions in case a new rule has to be evaluated
                            true_conds = true_conds + [cond]

                            purge_rule_ids = self._filter_rules(not_visited_rules, cond, is_right_cond=True)
                            for idx in purge_rule_ids:
                                not_visited_rules.pop(idx)

                if valid_rule:
                    satisfied_rules.append(not_visited_rules[0])
                    not_visited_rules.pop(0)

            if len(satisfied_rules) > 0:
                labels = [rule[self.LABEL] for rule in satisfied_rules]
                predicted_label = max(labels, key=lambda x:labels.count(x))

                predicted_labels.append(predicted_label)

                if log:
                    print('The found rules are:')
                    for rule in satisfied_rules:
                        self._print_rule(rule, None, False)
            else:
                if log:
                    print('\033[93mWARNING:\033[0m No rule has been found that satisfies the current sample conditions.')
                predicted_labels.append(-1)

            if log:
                if predicted_labels[-1] == groundtruth[row]:
                    print('\033[92mThe correct label is: ' + str(groundtruth[row]) + '\033[0m')
                else:
                    print('\033[91mThe correct label is: ' + str(groundtruth[row]) + '\033[0m')

        return predicted_labels

    def _filter_rules(self, rules, cond, is_right_cond):
        filtered_ids = []

        for r in range(len(rules)):
            # Find rules that do not satisfy the correct condition
            if is_right_cond:
                attr, values = zip(*rules[r][self.CONDITIONS])
                if cond[0] in attr and cond not in rules[r][self.CONDITIONS]:
                    filtered_ids.append(r)
                    continue
            else:
                # If false condition is present in the rule, discard it
                if cond in rules[r][self.CONDITIONS]:
                    filtered_ids.append(r)

        return sorted(filtered_ids, reverse=True)
