from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist


def bag_of_words(bestwords, words):
    return dict([(word, True) for word in words])


def bag_of_words_not_in_set(bestwords, words, badwords):
    return bag_of_words(bestwords, set(words) - set(badwords))


def bag_of_best_words(bestwords, words):
    return dict([(word, True) for word in words if word in bestwords])


def best_words(corp, limit=10000):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for word in corp.words(categories=['pos']):
        word_fd[word.lower()] += 1
        label_word_fd['pos'][word.lower()] += 1

    for word in corp.words(categories=['neg']):
        word_fd[word.lower()] += 1
        label_word_fd['neg'][word.lower()] += 1

    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    best = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:limit]
    bestwords = set([w for w, s in best])
    return bestwords


def bag_of_non_stopwords(bestwords, words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(bestwords, words, badwords)


def bag_of_bigrams_words(bestwords, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    words_list = [word for word in words]
    return bag_of_non_stopwords(bestwords, words_list + bigrams)


def bag_of_best_bigram_words(bestwords, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    bestwords_and_bigrams = dict([(bigram, True) for bigram in bigrams])
    bestwords_and_bigrams.update(bag_of_best_words(bestwords, words))
    return bestwords_and_bigrams