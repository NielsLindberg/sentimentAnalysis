import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
import csv
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC
import random
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

posdata = []
with open('positive-data.csv', 'rb') as myfile:
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        posdata.append(val[0])

negdata = []
with open('negative-data.csv', 'rb') as myfile:
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        negdata.append(val[0])


def word_split(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new


def word_split_sentiment(data):
    data_new = []
    for (word, sentiment) in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append((word_filter, sentiment))
    return data_new


def word_feats(words):
    return dict([(word, True) for word in words])


stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such',
                                                 'few', 'so', 'too', 'very', 'just', 'any', 'once'))


def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    """
    print words
    for ngram in itertools.chain(words, bigrams):
        if ngram not in stopset:
            print ngram
    exit()
    """
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def bigram_word_feats_stopwords(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    """
    print words
    for ngram in itertools.chain(words, bigrams):
        if ngram not in stopset:
            print ngram
    exit()
    """
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in stopset])


# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx):
    negfeats = [(featx(f), 'neg') for f in word_split(negdata)]
    posfeats = [(featx(f), 'pos') for f in word_split(posdata)]

    negcutoff = len(negfeats) * 3 / 4
    poscutoff = len(posfeats) * 3 / 4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    # using 3 classifiers
    classifier_list = ['nb', 'maxent', 'svm']

    for cl in classifier_list:
        if cl == 'maxent':
            classifierName = 'Maximum Entropy'
            classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, sparse=True,
                                                gaussian_prior_sigma=0, max_iter=1)
        elif cl == 'svm':
            classifierName = 'SVM'
            classifier = SklearnClassifier(LinearSVC(), sparse=False)
            classifier.train(trainfeats)
        else:
            classifierName = 'Naive Bayes'
            classifier = NaiveBayesClassifier.train(trainfeats)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

        accuracy = nltk.classify.util.accuracy(classifier, testfeats)
        pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
        pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
        pos_fmeasure = nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
        neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
        neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
        neg_fmeasure = nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

        print
        ''
        print
        '---------------------------------------'
        print
        'SINGLE FOLD RESULT ' + '(' + classifierName + ')'
        print
        '---------------------------------------'
        print
        'accuracy:', accuracy
        print
        'precision', (pos_precision + neg_precision) / 2
        print
        'recall', (pos_recall + neg_recall) / 2
        print
        'f-measure', (pos_fmeasure + neg_fmeasure) / 2

        # classifier.show_most_informative_features()

    print
    ''

    ## CROSS VALIDATION

    trainfeats = negfeats + posfeats

    # SHUFFLE TRAIN SET
    # As in cross validation, the test chunk might have only negative or only positive data
    random.shuffle(trainfeats)
    n = 5  # 5-fold cross-validation

    for cl in classifier_list:

        subset_size = len(trainfeats) / n
        accuracy = []
        pos_precision = []
        pos_recall = []
        neg_precision = []
        neg_recall = []
        pos_fmeasure = []
        neg_fmeasure = []
        cv_count = 1
        for i in range(n):
            testing_this_round = trainfeats[i * subset_size:][:subset_size]
            training_this_round = trainfeats[:i * subset_size] + trainfeats[(i + 1) * subset_size:]

            if cl == 'maxent':
                classifierName = 'Maximum Entropy'
                classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None,
                                                    sparse=True, gaussian_prior_sigma=0, max_iter=1)
            elif cl == 'svm':
                classifierName = 'SVM'
                classifier = SklearnClassifier(LinearSVC(), sparse=False)
                classifier.train(training_this_round)
            else:
                classifierName = 'Naive Bayes'
                classifier = NaiveBayesClassifier.train(training_this_round)

            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)
            for i, (feats, label) in enumerate(testing_this_round):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)

            cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
            cv_pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
            cv_pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
            cv_pos_fmeasure = nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
            cv_neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
            cv_neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
            cv_neg_fmeasure = nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

            accuracy.append(cv_accuracy)
            pos_precision.append(cv_pos_precision)
            pos_recall.append(cv_pos_recall)
            neg_precision.append(cv_neg_precision)
            neg_recall.append(cv_neg_recall)
            pos_fmeasure.append(cv_pos_fmeasure)
            neg_fmeasure.append(cv_neg_fmeasure)

            cv_count += 1

        print
        '---------------------------------------'
        print
        'N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')'
        print
        '---------------------------------------'
        print
        'accuracy:', sum(accuracy) / n
        print
        'precision', (sum(pos_precision) / n + sum(neg_precision) / n) / 2
        print
        'recall', (sum(pos_recall) / n + sum(neg_recall) / n) / 2
        print
        'f-measure', (sum(pos_fmeasure) / n + sum(neg_fmeasure) / n) / 2
        print
        ''


evaluate_classifier(word_feats)
# evaluate_classifier(stopword_filtered_word_feats)
# evaluate_classifier(bigram_word_feats)
# evaluate_classifier(bigram_word_feats_stopwords)