import collections
import nltk.classify.util
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
import tokinator
import csv
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

training_set = []


def word_extend(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.extend(word_filter)
    return data_new


def word_split(data):
    data_new = []
    for word in data:
            word_filter = [i.lower() for i in word.split()]
            data_new.append(word_filter)
    return data_new


def split_sets(data_path, text_column, sentiment_column):
    neudata = []
    posdata = []
    negdata = []
    alldata = []
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    limit = 10000

    with open(data_path, 'r', encoding='utf8') as csv_input:
        reader = csv.reader(csv_input, delimiter=',')
        next(reader)
        for row in reader:
            alldata.append(row[text_column])
            if row[sentiment_column] == 'Negative':
                negdata.append(row[text_column])
            elif row[sentiment_column] == 'Positive':
                posdata.append(row[text_column])
            elif row[sentiment_column] == 'Neutral':
                neudata.append(row[text_column])

    for word in word_extend(posdata):
        word_fd[word.lower()] += 1
        label_word_fd['pos'][word.lower()] += 1

    for word in word_extend(negdata):
        word_fd[word.lower()] += 1
        label_word_fd['neg'][word.lower()] += 1

    for word in word_extend(neudata):
        word_fd[word.lower()] += 1
        label_word_fd['neu'][word.lower()] += 1

    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    neu_word_count = label_word_fd['neu'].N()
    total_word_count = pos_word_count + neg_word_count + neu_word_count

    word_scores = {}

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                                               (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                                               (freq, neg_word_count), total_word_count)
        neu_score = BigramAssocMeasures.chi_sq(label_word_fd['neu'][word],
                                               (freq, neu_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score + neu_score

    best = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:limit]
    bestwords = set([w for w, s in best])
    return neudata, posdata, negdata, alldata, bestwords

def evaluate(feature_detector=tokinator.bag_of_best_bigram_words):

    data_path = 'D:/Workspace/CBS/BigSocialData/SentimentAnalysis/data/data.csv'
    neudata, posdata, negdata, alldata, topwords = split_sets(data_path, 0, 1)

    # topwords = tokinator.best_words(word_split(alldata), 10000)

    negfeats = [(feature_detector(topwords, f), 'neg') for f in word_split(negdata)]
    posfeats = [(feature_detector(topwords, f), 'pos') for f in word_split(posdata)]
    neufeats = [(feature_detector(topwords, f), 'neu') for f in word_split(neudata)]

    negcutoff = int(len(negfeats) * 3 / 4)
    poscutoff = int(len(posfeats) * 3 / 4)
    neucutoff = int(len(neufeats) * 3 / 4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + neufeats[:neucutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + neufeats[neucutoff:]

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
    print('neu precision:', precision(refsets['neu'], testsets['neu']))
    print('neu recall:', recall(refsets['neu'], testsets['neu']))
    classifier.show_most_informative_features(20)


methods = list([tokinator.bag_of_words, tokinator.bag_of_non_stopwords,
                tokinator.bag_of_bigrams_words, tokinator.bag_of_best_bigram_words,
                tokinator.bag_of_best_words])


for method in methods:
    evaluate(method)
