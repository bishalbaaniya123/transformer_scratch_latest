import unicodedata
import re
import math
import time
from io import open

from .vocabulary import Lang
from constants import *


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False, initialize=False):
    print("Reading lines...")

    lines1 = open(path + '/%s.txt' % lang1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(path + '/%s.txt' % lang2, encoding='utf-8').read().strip().split('\n')

    L1 = [normalize_string(s) for s in lines1]
    L2 = [normalize_string(s) for s in lines2]

    l1 = [i for i, e in enumerate(L1) if e == ' ']
    L11 = [j for i, j in enumerate(L1) if i not in l1]
    L22 = [j for i, j in enumerate(L2) if i not in l1]

    l2 = [i for i, e in enumerate(L22) if e == ' ']
    L1 = [j for i, j in enumerate(L11) if i not in l2]
    L2 = [j for i, j in enumerate(L22) if i not in l2]
    pairs = [list(x) for x in zip(L1, L2)]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    return pairs


def pre_process(lang1, lang2, reverse=False):
    print("Reading lines...")
    lan1 = 'english'
    lan2 = 'my'

    lines1 = open(path + '/%s.txt' % lan1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(path + '/%s.txt' % lan2, encoding='utf-8').read().strip().split('\n')

    L1 = [normalize_string(s) for s in lines1]
    L2 = [normalize_string(s) for s in lines2]

    l1 = [i for i, e in enumerate(L1) if e == ' ']
    L11 = [j for i, j in enumerate(L1) if i not in l1]
    L22 = [j for i, j in enumerate(L2) if i not in l1]

    l2 = [i for i, e in enumerate(L22) if e == ' ']
    L1 = [j for i, j in enumerate(L11) if i not in l2]
    L2 = [j for i, j in enumerate(L22) if i not in l2]
    pairs = [list(x) for x in zip(L1, L2)]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang


def filter_pair(p):
    lan_1 = len(p[0].split(' '))
    lan_2 = len(p[1].split(' '))
    return (lan_1 < MAX_LENGTH and lan_2 < MAX_LENGTH) and (lan_1 > 1 and lan_2 > 1)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    pairs = read_langs(lang1, lang2, reverse)
    input_lang, output_lang = pre_process(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# Preparing Training Data
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
