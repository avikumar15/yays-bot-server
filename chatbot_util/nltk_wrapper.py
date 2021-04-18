import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer, StemmerI

stemmer = PorterStemmer()


# nltk.download('punkt')

def tokenise(phrase):
    return nltk.word_tokenize(phrase)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenised_sentence, words):
    """
    ["I", "no", "where", "you", "we", "are", "no", "tonight", "go"]
    ["where", "are", "we", "go", "tonight"]
    [0, 0, 1, 0, 1, 1, 0, 1, 1]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenised_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

#
# phrase = "Where are we going, tonight?"
# print(phrase)
#
# tok = tokenise(phrase)
# print(tok)
#
# words = ["go", "went", "going", "goed"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)
#
# words = ["I", "no", "where", "you", "we", "are", "no", "tonight", "go"]
# sentence = ["where", "are", "we", "go", "tonight"]
#
# print(bag_of_words(sentence, words))
