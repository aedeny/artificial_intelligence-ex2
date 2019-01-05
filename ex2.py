import csv
import math
import operator
from collections import defaultdict
from typing import Tuple


def load_data(file_name):
    with open(file_name, 'r') as train_file:
        attributes = str(next(train_file)).strip().split('\t')
        reader = csv.DictReader(train_file, attributes, delimiter='\t')
        train_data = [row for row in reader]
    return train_data, attributes


def entropy(examples, target_att):
    values_to_occurrences = get_values_to_occurrences(examples, target_att)
    total = 0
    n = len(examples)
    for v in values_to_occurrences.items():
        p = v[1] / n
        total -= p * math.log2(p)
    return total


def get_values_to_occurrences(examples, attribute):
    values_to_occurrences = defaultdict(lambda: 0)
    for e in examples:
        values_to_occurrences[e[attribute]] += 1
    return values_to_occurrences


class Tree:

    def __init__(self, attribute, children=None):
        if children is None:
            children = {}
        self.attribute = attribute
        self.children = children

    def __repr__(self):
        return '{}={}'.format(self.attribute, self.children.keys())

    def __str__(self):
        return self._to_string(0)

    def _to_string(self, depth):
        tabs = '\t' * depth + '|' * int(depth != 0)
        s = ''
        item: Tuple[str, Tree]
        for item in self.children.items():
            s += '{}{}={}'.format(tabs, self.attribute, item[0])
            if item[1].children == {}:
                s += ':{}\n'.format(item[1].attribute)
            else:
                s += '\n{}'.format(item[1]._to_string(depth + 1))

        return s

    def trim(self):
        if self.children == {}:
            return self.attribute
        v = next(iter(self.children.values())).attribute
        for value in self.children.values():
            temp = value.trim()
            if temp != v:
                return self
        self.attribute = v
        self.children = {}

        return self

    def predict(self, example: dict):
        while self.children != {}:
            return self.children[example[self.attribute]].predict(example)
        return self.attribute


class DecisionTree:
    _tree: Tree

    def __init__(self, file_name):
        self._train_data, self.attributes = load_data(file_name)
        self._target_att = self.attributes[-1]
        self._tree = self._dtl_top_level()

    def __str__(self):
        return str(self._tree)

    def predict(self, example: dict):
        return self._tree.predict(example)

    def _get_data(self, attributes_to_values_query):
        """
        Returns a list of entries satisfying query.
        :type attributes_to_values_query: dict
        """
        result = []
        for entry in self._train_data:
            for item in attributes_to_values_query.items():
                if entry[item[0]] != item[1]:
                    break
                result.append(entry)

        return result

    def _most_common_class(self, examples):
        """
        Returns the most common class among the given examples.
        :type examples: list
        :param examples: Examples
        :return: The most common class among the examples.
        """
        values_to_occurrences = get_values_to_occurrences(examples, self._target_att)
        return max(values_to_occurrences.items(), key=operator.itemgetter(1))[0]

    def _dtl(self, examples: list, attributes: list, default: Tree) -> Tree:
        """
        Creates a decision tree recursively.
        :param examples: A list of dicts, where each dict is in the form {attribute : value}.
        :param attributes: A list of attributes.
        :param default: A Tree to return in case there are no examples.
        :return: A decision tree.
        """
        if not examples:
            return default

        # If all examples have the same class
        values_to_occurrences = get_values_to_occurrences(examples, self._target_att)
        if len(values_to_occurrences) == 1:
            return Tree(next(iter(values_to_occurrences)))

        if not attributes:
            return Tree(self._most_common_class(examples))

        best_att = self._choose_attribute(attributes, examples)
        tree = Tree(best_att)
        best_att_values = {e[best_att] for e in examples}

        for v in sorted(list(best_att_values)):
            examples_v = [e for e in examples if e[best_att] == v]
            sub_tree = self._dtl(examples_v, list(set(attributes) - {best_att}),
                                 Tree(self._most_common_class(examples)))
            tree.children[v] = sub_tree

        return tree.trim()

    def _choose_attribute(self, attributes: list, examples: list):
        att_to_ig = {attribute: self._information_gain(examples, attribute) for attribute in attributes}
        return max(att_to_ig.items(), key=operator.itemgetter(1))[0]

    def _information_gain(self, examples, attribute):
        ent = entropy(examples, self._target_att)
        values_to_occurrences = get_values_to_occurrences(examples, attribute)
        k = len(examples)
        s = 0
        for value, occurrences in values_to_occurrences.items():
            s += occurrences / k * entropy(self._get_data({attribute: value}), self._target_att)
        return ent - s

    def _dtl_top_level(self):
        attributes = self.attributes[:-1]
        default = Tree(self._most_common_class(self._train_data))
        return self._dtl(self._train_data, attributes, default)


if __name__ == '__main__':
    decision_tree = DecisionTree('data/train.txt')
    print(decision_tree)

    my_example = {'sex': 'female', 'pclass': '3rd', 'age': 'child'}
    print('Prediction for example {} is {}.'.format(my_example, decision_tree.predict(my_example)))
    pass
