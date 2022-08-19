

import struct



class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = {}
        self._str_to_label = {}
        self._size = 0
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if line[0:2] == '\\#':
                        line = '#\n'
                    elif line[0] == '#':
                        continue
                    self._label_to_str[self._size] = line[:-1] # remove the line ending
                    self._str_to_label[line[:-1]] = self._size
                    self._size += 1

    def _string_from_label(self, label):
        return self._label_to_str[label]

    def _label_from_string(self, string):
        try:
            return self._str_to_label[string]
        except KeyError as e:
            raise KeyError(
                'ERROR: Your transcripts contain characters (e.g. \'{}\') which do not occur in \'{}\'! Use ' \
                'util/check_characters.py to see what characters are in your [train,dev,test].csv transcripts, and ' \
                'then add all these to \'{}\'.'.format(string, self._config_file, self._config_file)
            ).with_traceback(e.__traceback__)

    def has_char(self, char):
        return char in self._str_to_label

    def encode(self, string):
        res = []
        for char in string:
            res.append(self._label_from_string(char))
        return res

    def decode(self, labels):
        res = ''
        for label in labels:
            res += self._string_from_label(label)
        return res

    def serialize(self):
        # Serialization format is a sequence of (key, value) pairs, where key is
        # a uint16_t and value is a uint16_t length followed by `length` UTF-8
        # encoded bytes with the label.
        res = bytearray()

        # We start by writing the number of pairs in the buffer as uint16_t.
        res += struct.pack('<H', self._size)
        for key, value in self._label_to_str.items():
            value = value.encode('utf-8')
            # struct.pack only takes fixed length strings/buffers, so we have to
            # construct the correct format string with the length of the encoded
            # label.
            res += struct.pack('<HH{}s'.format(len(value)), key, len(value), value)
        return bytes(res)

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file




def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def wer(ground_truth, prediction):    
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    wer = word_distance/word_length
    wer = min(wer, 1.0)
    return wer
    
def cer(ground_truth, prediction):
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    cer=char_distance/char_length
    cer = min(cer, 1.0)
    return  cer