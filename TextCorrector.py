import numpy as np
import re
from operator import itemgetter


def intersect(x, y):
    """
    Whether two segments intersect
    """
    return x[1] > y[0] and x[0] < y[1]


def distance(x, y):
    """
    Compare if two characters are equal or not
    :param string x: a single character
    :param string y: a single character
    :return: 1 if x and y not equal, 0 if x and y equal
    :rtype: int
    """
    if set((x, y)).issubset({' ', '\n'}):
        return 0
    return int(x.lower() != y.lower())


class TextCorrector:

    def __init__(self, threshold=0.3):
        """
        Initializes the class.
        :param threshold: the threshold we choose for the correction of a word
        """
        self.threshold = threshold
        self.words = []
        self.changed = []

    def __str__(self):
        """
        String representation of the class
        """
        return f"{self.__class__.__name__}(threshold={self.threshold})"

    def load(self, input_words):
        """
        Loads the list of words that will be used for correcting
        :param input_words: a path to the txt file containing the list of words, a list of words or a set of words
        """
        if isinstance(input_words, list):
            words = list(set(input_words))
        elif isinstance(input_words, set):
            words = list(input_words)
        elif isinstance(input_words, str):
            with open(input_words, 'r') as file:
                words = list(set(file.read().splitlines()))
        else:
            raise TypeError('Input type is either a list, set, or the path of a file')
        self.words = sorted(words, key=len, reverse=True)

    def forward(self, text, query):
        """
        The forward step in the dynamic programming
        """
        m = len(query)
        n = len(text)

        E = np.zeros((m + 1, n + 1))
        D = {(0, i): [2] for i in range(n + 1)}
        length = len(query)

        for i in range(1, m + 1):
            E[i, 0] = i
            D[(i, 0)] = [0]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                choices = [
                    E[i - 1, j] + 1,
                    E[i - 1, j - 1] + distance(query[i - 1], text[j - 1]),
                    E[i, j - 1] + 1
                ]
                val = min(choices)
                D[(i, j)] = np.where(choices == val)[0].tolist()
                E[i, j] = val
        return D, [(i, E[m][i] / length) for i in range(n + 1) if E[m][i] / length <= self.threshold]

    def backward(self, D, start):
        """
        The backward step in the dynamic programming
        """
        if not start[0]:
            return {start}
        res = set()
        if 0 in D[start]:
            res.add((start[0] - 1, start[1]))
        if 1 in D[start]:
            res.add((start[0] - 1, start[1] - 1))
        if 2 in D[start]:
            res.add((start[0], start[1] - 1))
        return set().union(*[self.backward(D, x) for x in res])

    def find(self, text, query):
        """
        Find the substrings in text that look the most like **query**, with the distance to query.
        """
        D, E = self.forward(text, query)
        m = len(query)
        start = [((m, x[0]), x[1]) for x in E]
        result = set(((y[1], x[0][1]), x[1]) for x in start for y in self.backward(D, x[0]))
        return result

    def choose_positions(self, text, query):
        """
        Chooses the positions to correct, making sure there are no overlaps with the already corrected text.
        """
        positions = sorted(self.find(text, query), key=itemgetter(1))
        chosen_positions = []
        for position in positions:
            for chosen_position in chosen_positions + self.changed:
                if intersect(position[0], chosen_position):
                    break
            else:
                chosen_positions.append(position[0])
        return sorted(chosen_positions)

    def add_positions(self, positions, shift):
        """
        Changes the indices of the saved positions as the text changes, and adds them to the list
        """
        if not positions:
            return
        if not self.changed:
            position = positions[0]
            self.changed.append((position[0], position[0] + shift))
            cum_shift = shift - position[1] + position[0]
            i_start = 1
        else:
            cum_shift = 0
            i_start = 0
        for position in positions[not self.changed:]:
            for i in range(i_start, len(self.changed)):
                if self.changed[i][0] > position[0]:
                    self.changed.insert(i, (position[0] + cum_shift, position[0] + cum_shift + shift))
                    cum_shift += shift - position[1] + position[0]
                    i_start = i + 1
                    break
                else:
                    self.changed[i] = (self.changed[i][0] + cum_shift, self.changed[i][1] + cum_shift)
            else:
                self.changed.append((position[0] + cum_shift, position[0] + cum_shift + shift))
                cum_shift += shift - position[1] + position[0]
                i_start = len(self.changed)
        for i in range(i_start, len(self.changed)):
            self.changed[i] = (self.changed[i][0] + cum_shift, self.changed[i][1] + cum_shift)

    def correct(self, text):
        """Correct the text"""
        self.changed = []
        corrected_text = text
        for query in self.words:
            length = len(corrected_text)
            chosen_positions = self.choose_positions(corrected_text, query)
            self.add_positions(chosen_positions, len(query) + 2)
            selectors = [(0, 0)] + chosen_positions + [(length, length)]
            spaced_query = " " + query + " "
            corrected_text = spaced_query.join([corrected_text[x[1]:y[0]] for x, y in zip(selectors, selectors[1:])])
        result = re.sub(' +', ' ', corrected_text)
        return result
