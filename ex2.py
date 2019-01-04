import csv
import math
import operator
from collections import defaultdict


class Tree:
    def __init__(self, attribute, children=None):
        if children is None:
            children = {}
        self.attribute = attribute
        self.children = children

    def __repr__(self):
        return '{}={}'.format(self.attribute, self.children.keys())

    def to_string(self, depth):
        tabs = '\t' * depth + '|' * int(depth != 0)
        s = ''
        for item in self.children.items():
            s += '{}{}={}'.format(tabs, self.attribute, item[0])

            if item[1].children == {}:
                s += ':{}\n'.format(item[1].attribute)
            else:
                s += '\n{}'.format(item[1].to_string(depth + 1))
        return s

    def trim(self):
        if self.children == {}:
            return self.attribute
        v = next(iter(self.children.values())).attribute
        value: Tree
        for value in self.children.values():
            temp = value.trim()
            if temp != v:
                return self
        self.attribute = v
        self.children = {}

        return self


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
        """

        :type default: Tree
        """
        if not examples:
            return default

        # If all examples have the same class
        values_to_occurrences = self._get_values_to_occurrences(examples, self.target_att)
        if len(values_to_occurrences) == 1:
            return Tree(next(iter(values_to_occurrences)))

        if not attributes:
            return Tree(self._mode(examples))

        best_att = self._choose_attribute(attributes, examples)
        tree = Tree(best_att)
        best_att_values = {e[best_att] for e in examples}

        for v in sorted(list(best_att_values)):
            examples_v = [e for e in examples if e[best_att] == v]
            sub_tree = self._dtl(examples_v, list(set(attributes) - {best_att}), Tree(self._mode(examples)))
            tree.children[v] = sub_tree

        return tree.trim()

    def _choose_attribute(self, attributes, examples):
        att_to_ig = {attribute: self._information_gain(examples, attribute) for attribute in attributes}
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
        default = Tree(self._mode(self.train_data))
        return self._dtl(self.train_data, attributes, default)


if __name__ == '__main__':
    model = Model('data/train.txt')
    decision_tree = model.dtl_top_level()
    print(decision_tree.to_string(0))
    # model._reduce_tree(decision_tree)
    pass
