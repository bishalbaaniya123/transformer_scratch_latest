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
        if unicodedata.category(c) != 'Mn'  # Mn is the Unicode category for non-spacing marks
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)  # adding space before punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # removing any non-letter or punctuation character
    return s


"""
This function is used to read and process parallel text data from two different languages, specified by the variables 
"lang1" and "lang2". The function takes four parameters: "lang1" and "lang2" are the names of the two languages being
 read and processed, "reverse" is a boolean value that determines whether the language pairs are reversed 
 (i.e. if the first language is the second language and vice versa), and "initialize" is a boolean value that is not 
 used in this function.

The function starts by printing "Reading lines..." to the console to indicate that the reading process has begun. Next, 
the function uses the "open()" function to open two text files, one for each language, located in a specified path. The
encoding for the files is set to 'utf-8'. The contents of the files are then read using the "read()" method, and the
resulting strings are stripped of any leading or trailing white spaces using the "strip()" method. The strings are then 
split into individual lines using the "split('\n')" method.

The next step is to normalize the text in each language using the "normalize_string(s)" function. This function is not 
defined in this code snippet, so it is unclear exactly what it does. However, it can be assumed that it is used to 
process the text in some way, such as removing special characters or converting all text to lowercase. The normalized 
text for each language is then stored in separate lists, "L1" and "L2".

The function then creates two new lists, "l1" and "l2", by using list comprehension to find all the indexes of empty 
strings in "L1" and "L2" respectively. Then using these indexes, it removes the empty strings from "L1" and "L2" by 
creating two new lists "L11" and "L22" respectively.

Finally, the function creates a list of language pairs by using the "zip()" function to combine the processed text from 
"L1" and "L2" into pairs. If the "reverse" parameter is set to True, the function will use a list comprehension to 
reverse the order of the language pairs using the "reversed()" function. The resulting list of language pairs is then 
returned by the function.
"""


def read_langs(lang1, lang2, reverse=False, initialize=False):
    print("Reading lines...")

    # Open the first language file and read the contents, then split the contents into a list of lines
    lines1 = open(path + '/%s.txt' % lang1, encoding='utf-8').read().strip().split('\n')

    # Open the second language file and read the contents, then split the contents into a list of lines
    lines2 = open(path + '/%s.txt' % lang2, encoding='utf-8').read().strip().split('\n')

    # Normalize the first language lines
    L1 = [normalize_string(s) for s in lines1]
    # Normalize the second language lines
    L2 = [normalize_string(s) for s in lines2]

    # Get the index of the lines with only whitespace in the first language
    l1 = [i for i, e in enumerate(L1) if e == ' ']

    # Remove the lines with only whitespace from the first language
    L11 = [j for i, j in enumerate(L1) if i not in l1]
    L22 = [j for i, j in enumerate(L2) if i not in l1]

    # Get the index of the lines with only whitespace in the second language
    l2 = [i for i, e in enumerate(L22) if e == ' ']

    # Remove the lines with only whitespace from the second language
    L1 = [j for i, j in enumerate(L11) if i not in l2]
    L2 = [j for i, j in enumerate(L22) if i not in l2]

    # Create pairs of sentences from the two languages
    pairs = [list(x) for x in zip(L1, L2)]

    # If the reverse flag is set, reverse the order of the language pairs
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    return pairs


"""
This function is used for pre-processing the data before it is passed to the model. The function takes in two arguments
 - lang1 and lang2, which are the languages that we want to translate between. The function also takes an optional 
 argument reverse which is set to False by default.

The function starts by printing "Reading lines..." to indicate that the function has started.

Then it assigns variable lan1 and lan2 with 'english' and 'maya' respectively.

Then it opens the text file for the first language specified in the variable lan1 and reads the contents, stripping off
any leading or trailing whitespaces and then splits the text by newline character '\n' and assigns it to the variable 
lines1. Similarly, it does the same for the second language and assigns it to the variable lines2.

Next, it normalizes the strings in the list lines1 and assigns it to L1 and normalizes the strings in the list lines2 
and assigns it to L2.

Then it creates a list of indices where ' ' is present in the list L1. From that list L11 is created by keeping the 
elements of L1 where the index is not present in the list of indices created before. Similarly, L22 is created by 
keeping the elements of L2 where the index is not present in the list of indices created before.

Then it creates another list of indices where ' ' is present in the list L22. From that list L1 is created by keeping 
the elements of L11 where the index is not present in the list of indices created before. Similarly, L2 is created by 
keeping the elements of L22 where the index is not present in the list of indices created before.

Then it creates a list of pairs by zipping L1 and L2 and assigns it to the variable pairs.

If the reverse variable is True, the function reverses the order of elements in each pair of the list pairs.

Then it creates two Lang objects, input_lang and output_lang, with the languages specified in lang1 and lang2 
respectively.

Finally, the function returns the input_lang and output_lang objects.
"""


def pre_process(lang1, lang2, reverse=False):
    print("Reading lines...")
    lan1 = 'english'
    lan2 = 'my'

    lines1 = open(path + '/%s.txt' % lan1, encoding='utf-8').read().strip().split(
        '\n')  # open the file of language1, read the lines and split them
    lines2 = open(path + '/%s.txt' % lan2, encoding='utf-8').read().strip().split(
        '\n')  # open the file of language2, read the lines and split them

    L1 = [normalize_string(s) for s in lines1]  # apply normalize_string function to each sentence of language1
    L2 = [normalize_string(s) for s in lines2]  # apply normalize_string function to each sentence of language2

    l1 = [i for i, e in enumerate(L1) if e == ' ']  # get the index of elements with ' ' in L1
    L11 = [j for i, j in enumerate(L1) if i not in l1]  # get all elements of L1 which are not in l1
    L22 = [j for i, j in enumerate(L2) if i not in l1]  # get all elements of L2 which are not in l1

    l2 = [i for i, e in enumerate(L22) if e == ' ']  # get the index of elements with ' ' in L22
    L1 = [j for i, j in enumerate(L11) if i not in l2]  # get all elements of L11 which are not in l2
    L2 = [j for i, j in enumerate(L22) if i not in l2]  # get all elements of L22 which are not in l2
    pairs = [list(x) for x in zip(L1, L2)]  # zip L1 and L2 together and make a list of them

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)  # if reverse is true, then input_lang becomes lang2
        output_lang = Lang(lang1)  # if reverse is true, then output_lang becomes lang1
    else:
        input_lang = Lang(lang1)  # if reverse is false, then input_lang becomes lang1
        output_lang = Lang(lang2)  # if reverse is false, then output_lang becomes lang2

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
