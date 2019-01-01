import csv

import numpy as np


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

    def _entropy(self, values):
        total = 0
        for v in values:
            p = v / sum(values)
            total -= p * np.log2(p)
        return total

    def _entropy_tx(self, target_att, attribute):
        value_to_occurences = {}
        s_pos = self.get_data({target_att: True})
        s_neg = self.get_data({target_att: False})

        pass

    def gain(self, target, attributes):

        return self._entropy(target) - self._entropy_tx(target, attributes)

    def id3(self, examples, target_att, attributes):
        root = Node()
        positive_count = len([x for x in examples if x[target_att] == 'yes'])
        size = len(examples)
        if positive_count == len(examples):
            return Node(target_att, True)
        elif positive_count == 0:
            return Node(target_att, False)
        elif not attributes:
            pass

        attribute_to_gain = {}
        for attribute in attributes:
            attribute_to_gain[attribute] = self.gain(target_att, attribute)


if __name__ == '__main__':
    model = Model('data/train.txt')
    model.id3(model.train_data, model.attributes[-1], model.attributes[:-1])
