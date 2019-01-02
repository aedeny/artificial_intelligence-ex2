import csv
import math
import operator
from collections import defaultdict


class Node:
    def __init__(self, attribute=None, value=None):
        self.attribute = attribute
        self.value = value
        self.children = []

    def __repr__(self):
        return '{}->{}'.format(self.attribute, self.value)


def _load_data(file_name):
    with open(file_name, 'r') as train_file:
        attributes = str(next(train_file)).strip().split('\t')
        reader = csv.DictReader(train_file, attributes, delimiter='\t')
        train_data = [row for row in reader]
    return train_data, attributes


class Model:
    def __init__(self, file_name):
        self.train_data, self.attributes = _load_data(file_name)
        self.target_att = self.attributes[-1]

    def get_data(self, attributes_to_values_query):
        """
        Returns a list of entries satisfying query.
        :type attributes_to_values_query: dict
        """
        result = []
        for entry in self.train_data:
            for item in attributes_to_values_query.items():
                if entry[item[0]] != item[1]:
                    break
                result.append(entry)

        return result

    @staticmethod
    def _get_values_to_occurrences(examples, attribute):
        values_to_occurrences = defaultdict(lambda: 0)
        for e in examples:
            values_to_occurrences[e[attribute]] += 1
        return values_to_occurrences

    def _entropy(self, examples, target_att):
        values_to_occurrences = self._get_values_to_occurrences(examples, target_att)
        total = 0
        n = len(examples)
        for v in values_to_occurrences.items():
            p = v[1] / n
            total -= p * math.log2(p)
        return total

    def _mode(self, examples):
        """
        Returns the most common class among the examples.
        :type examples: list
        :param examples: Examples
        :return: The most common class among the examples.
        """
        values_to_occurrences = self._get_values_to_occurrences(examples, self.target_att)
        return max(values_to_occurrences.items(), key=operator.itemgetter(1))[0]

    def _dtl(self, examples, attributes, default):
        if not examples:
            return default

        # If all examples have the same class
        values_to_occurrences = self._get_values_to_occurrences(examples, self.target_att)
        if len(values_to_occurrences) == 1:
            return Node(next(iter(values_to_occurrences)), examples[0][self.target_att])

        if not attributes:
            return Node(self.target_att, self._mode(examples))

        best_att = self._choose_attribute(attributes, examples)
        tree = Node(best_att)
        best_att_values = {e[best_att] for e in examples}

        for v in best_att_values:
            examples_v = []
            for ei in examples:
                if ei[best_att] == v:
                    examples_v.append(ei)
            sub_tree = self._dtl(examples_v, list(set(attributes) - {best_att}), self._mode(examples))
            tree.children.append(Node(sub_tree, v))
        return tree

    def _choose_attribute(self, attributes, examples):
        att_to_ig = {}
        for attribute in attributes:
            att_to_ig[attribute] = self._information_gain(examples, attribute)
        return max(att_to_ig.items(), key=operator.itemgetter(1))[0]

    def _information_gain(self, examples, attribute):
        entropy = self._entropy(examples, self.target_att)
        values_to_occurrences = self._get_values_to_occurrences(examples, attribute)
        k = len(examples)
        s = 0
        for value, occurrences in values_to_occurrences.items():
            s += occurrences / k * self._entropy(self.get_data({attribute: value}), self.target_att)
        return entropy - s

    def dtl_top_level(self):
        attributes = self.attributes[:-1]
        default = Node(self.target_att, self._mode(self.train_data))
        return self._dtl(self.train_data, attributes, default)


if __name__ == '__main__':
    model = Model('data/train.txt')
    decision_tree = model.dtl_top_level()
    pass
