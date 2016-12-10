from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures


def bag_of_words(words):
    # returns a dictionary with all the words and true
    return dict([(word, True) for word in words])


def bag_of_words_not_in_set(words, bad_words):
    # returns a call to bag of words where a set of words have been substracted from another
    # used to remove a set of stopwords from a full set of words
    return bag_of_words(set(words) - set(bad_words))


def bag_of_best_words(best_words, words):
    # returns a dictionary with words and true if the word exists in a parsed list of best_words
    return dict([(word, True) for word in words if word in best_words])


def bag_of_non_stopwords(words, stop_file='english'):
    # returns a call to the bag of words not in set parsing in the full list of words
    # together with the list of english stop words.
    bad_words = stopwords.words(stop_file)
    return bag_of_words_not_in_set(words, bad_words)


def bag_of_bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    # returns a call to bag of non stop words with a list of the best 200 bigrams
    # together with all the singular words
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    words_list = [word for word in words]
    return bag_of_non_stopwords(words_list + bigrams)


def bag_of_best_bigram_words(best_words, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    # returns a call to bag of non stop words with a list of the best 200 bigrams
    # together with all the singular words that are also in the best_words list.
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    best_words_and_bigrams = dict([(bigram, True) for bigram in bigrams])
    best_words_and_bigrams.update(bag_of_best_words(best_words, words))
    return bag_of_non_stopwords(best_words_and_bigrams)