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
    tokenised_sentence = [stem(w) for w in tokenised_sentence]
    bag = [float(1) if w in tokenised_sentence else float(0) for w in words]
    return bag


phrase = "Where are we going, tonight?"
print(phrase)

tok = tokenise(phrase)
print(tok)

words = ["go", "went", "going", "goed"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

words = ["I", "no", "where", "you", "we", "are", "no", "tonight", "go"]
sentence = ["where", "are", "we", "go", "tonight"]

print(bag_of_words(sentence, words))
