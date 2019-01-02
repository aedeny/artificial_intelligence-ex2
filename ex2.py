import csv
import math
import operator
from collections import defaultdict


class Node:
    def __init__(self, attribute=None, value=None):
        self.attribute = attribute
        self.value = value
        self.children = []


def _load_data(file_name):
    with open(file_name, 'r') as train_file:
        attributes = str(next(train_file)).strip().split('\t')
        reader = csv.DictReader(train_file, attributes, delimiter='\t')
        train_data = [row for row in reader]
    return train_data, attributes


class Model:
    def __init__(self, file_name):
        self.train_data, self.attributes = _load_data(file_name)

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
    def _entropy(values):
        total = 0
        for v in values:
            p = v / sum(values)
            total -= p * math.log2(p)
        return total

    def gain(self, target, attributes):
        return self._entropy(target) - self._entropy_tx(target, attributes)

    def _mode(self, examples):
        """
        Returns the most common class among the examples.
        :type examples: list
        :param examples: Examples
        :return: The most common class among the examples.
        """
        values_to_occurrences = defaultdict(lambda: 0)
        for e in examples:
            values_to_occurrences[e[-1]] += 1
        return max(values_to_occurrences.items(), key=operator.itemgetter(1))[0]

    def id3(self, examples, attributes, default):
        target_att = attributes[-1]
        if not examples:
            return default

        # If all examples have the same class
        values_to_occurrences = defaultdict(lambda: 0)
        for e in examples:
            values_to_occurrences[e[target_att]] += 1
        if len(values_to_occurrences) == 1:
            return Node(next(iter(values_to_occurrences)), examples[target_att])

        if not attributes:
            return Node(target_att, self._mode(examples))

        best_att = self._choose_attribute(attributes, examples)
        tree = Node(best_att)
        best_att_values = {e[best_att] for e in examples}
        for v in best_att_values:
            examples_v = {e for e in examples if e[best_att] == v}
            sub_tree = self.id3(examples_v, list(set(attributes) - {best_att}), self._mode(examples))
            tree.children.append(Node(sub_tree, v))
        return tree

    def _choose_attribute(self, attributes, examples):
        pass


if __name__ == '__main__':
    model = Model('data/train.txt')
    model.id3(model.train_data, model.attributes[-1], model.attributes[:-1])
