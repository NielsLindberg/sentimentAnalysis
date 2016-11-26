import collections
import nltk.classify.util
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import csv
import tokinator
import language_detection
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
import os

training_set = []

def evaluate(corp, feature_detector=tokinator.bag_of_best_bigram_words):
    negids = corp.fileids('neg')
    posids = corp.fileids('pos')
    topwords = tokinator.best_words(corp, 10000)

    negfeats = [(feature_detector(topwords, corp.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(feature_detector(topwords, corp.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = int(len(negfeats) * 3 / 4)
    poscutoff = int(len(posfeats) * 3 / 4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', precision(refsets['pos'], testsets['pos']))
    print('pos recall:', recall(refsets['pos'], testsets['pos']))
    print('neg precision:', precision(refsets['neg'], testsets['neg']))
    print('neg recall:', recall(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features(20)


methods = list([tokinator.bag_of_words, tokinator.bag_of_non_stopwords,
                tokinator.bag_of_bigrams_words, tokinator.bag_of_best_bigram_words,
                tokinator.bag_of_best_words])

current_dir = os.getcwd()
os.chdir('D:/nltk_data/corpora/moviez')

reader = CategorizedPlaintextCorpusReader('.', r'.*\.txt',
                                          cat_pattern=r'(\w+)/*')
for method in methods:
    evaluate(reader, method)

os.chdir(current_dir)
