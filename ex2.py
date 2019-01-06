import csv
import math
import operator
from collections import defaultdict
from typing import Tuple


def load_data(file_name):
    with open(file_name, 'r') as f:
        attributes = str(next(f)).strip().split('\t')
        reader = csv.DictReader(f, attributes, delimiter='\t')
        data = [row for row in reader]
    return data, attributes


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


def most_common_class(examples, target_attribute):
    """
    Returns the most common class among the given examples.
    :type examples: list
    :param examples: Examples
    :param target_attribute:
    :return: The most common class among the examples.
    """
    values_to_occurrences = get_values_to_occurrences(examples, target_attribute)
    return max(values_to_occurrences.items(), key=operator.itemgetter(1))[0]


def get_data(attributes_to_values_query, train_data):
    """
    Returns a list of entries satisfying query.
    :param train_data:
    :type attributes_to_values_query: dict
    """
    result = []
    for entry in train_data:
        valid = True
        for item in attributes_to_values_query.items():
            if entry[item[0]] != item[1]:
                valid = False
                break
        if valid:
            result.append(entry)

    return result


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

    def __init__(self, train_data):
        self.name = 'DT'
        self._train_data, self.attributes = train_data
        self._target_att = self.attributes[-1]
        self._tree = self._dtl_top_level()

    def __str__(self):
        return str(self._tree)

    def predict(self, example: dict):
        return self._tree.predict(example)

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
            return Tree(most_common_class(examples, self._target_att))

        best_att = self._choose_attribute(attributes, examples)
        tree = Tree(best_att)
        best_att_values = {e[best_att] for e in examples}

        for v in sorted(list(best_att_values)):
            examples_v = [e for e in examples if e[best_att] == v]
            sub_tree = self._dtl(examples_v, list(set(attributes) - {best_att}),
                                 Tree(most_common_class(examples, self._target_att)))
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
            s += occurrences / k * entropy(get_data({attribute: value}, self._train_data), self._target_att)
        return ent - s

    def _dtl_top_level(self):
        attributes = self.attributes[:-1]
        default = Tree(most_common_class(self._train_data, self._target_att))
        return self._dtl(self._train_data, attributes, default)


class KNN:
    def __init__(self, train_data, k=5):
        self.name = 'KNN'
        self._train_data, self.attributes = train_data
        self._k = k

    def _distance(self, e1, e2) -> int:
        return sum([1 for attribute in self.attributes[:-1] if e1[attribute] != e2[attribute]])

    def predict(self, example) -> str:
        distances = sorted([(self._distance(e, example), e) for e in self._train_data], key=lambda x: x[0])[:self._k]
        knn = [distance[1] for distance in distances]
        return most_common_class(knn, self.attributes[-1])


class NaiveBayes:
    def __init__(self, train_data):
        self.name = 'naiveBayes'
        self.train_data, self.arguments = train_data

    def predict(self, example: dict):
        target_argument = self.arguments[-1]
        target_label_occurrences = get_values_to_occurrences(self.train_data, target_argument)
        probabilities = {t: {} for t in target_label_occurrences}
        n = len(self.train_data)

        # Trains
        for target_label in target_label_occurrences.items():
            for item in example.items():
                query = {target_argument: target_label[0], item[0]: item[1]}
                data_count = len(get_data(query, self.train_data))
                probabilities[target_label[0]][item[1]] = data_count / target_label[1]

        # Predicts
        result = {t: 1 for t in target_label_occurrences}
        for item in probabilities.items():
            for value in item[1].values():
                result[item[0]] *= value
            result[item[0]] *= target_label_occurrences[item[0]] / n

        return max(result.items(), key=operator.itemgetter(1))[0]


def test(test_data, model):
    results = [model.predict(e) for e in test_data[0]]
    accuracy = sum([1 for t, prediction in zip(test_data[0], results) if t[test_data[1][-1]] == prediction]) / len(
        results)
    return results, accuracy


if __name__ == '__main__':
    my_train_data = load_data('data/train.txt')
    my_test_data = load_data('data/test.txt')
    my_example = {'sex': 'female', 'pclass': '3rd', 'age': 'child'}

    models = [DecisionTree(my_train_data), KNN(my_train_data), NaiveBayes(my_train_data)]

    model_to_results = {}
    for model in models:
        predictions, accuracy = test(my_test_data, model)
        model_to_results[model.name] = {'predictions': predictions, 'accuracy': accuracy}

    with open('data/my_output.txt', 'w') as f:
        for i, models in enumerate(zip(x for x in model_to_results.items())):
            pass

