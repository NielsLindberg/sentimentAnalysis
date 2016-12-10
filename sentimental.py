import collections
import nltk.classify.util
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
import tokinator
import csv
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist


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


def split_sets(data_path, text_column, sentiment_column, limit):
    neudata = []
    posdata = []
    negdata = []
    alldata = []
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    with open(data_path, 'r', encoding='utf8') as csv_input:
        reader = csv.reader(csv_input, delimiter=';')
        next(reader)
        for row in reader:
            alldata.append(row[text_column])
            # Negative
            if row[sentiment_column] == '2':
                negdata.append(row[text_column])
            # Positive
            elif row[sentiment_column] == '1':
                posdata.append(row[text_column])
            # Neutral
            elif row[sentiment_column] == '0':
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
    best_words = set([w for w, s in best])
    return neudata, posdata, negdata, alldata, best_words


def evaluate(feature_name, feature_detector, use_best_words, limit):

    data_path = 'D:/Workspace/CBS/BigSocialData/SentimentAnalysis/data/all_text_actions_test_fix.csv'
    neudata, posdata, negdata, alldata, best_words = split_sets(data_path, 21, 23, limit)

    if use_best_words:
        negfeats = [(feature_detector(best_words, f), 'neg') for f in word_split(negdata)]
        posfeats = [(feature_detector(best_words, f), 'pos') for f in word_split(posdata)]
        neufeats = [(feature_detector(best_words, f), 'neu') for f in word_split(neudata)]
    else:
        negfeats = [(feature_detector(f), 'neg') for f in word_split(negdata)]
        posfeats = [(feature_detector(f), 'pos') for f in word_split(posdata)]
        neufeats = [(feature_detector(f), 'neu') for f in word_split(neudata)]

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

    accuracy = nltk.classify.util.accuracy(classifier, testfeats)
    pos_precision = precision(refsets['pos'], testsets['pos'])
    pos_recall = recall(refsets['pos'], testsets['pos'])
    neg_precision = precision(refsets['neg'], testsets['neg'])
    neg_recall = recall(refsets['pos'], testsets['pos'])
    neu_precision = precision(refsets['neu'], testsets['neu'])
    neu_recall = recall(refsets['neu'], testsets['neu'])

    classifier_with_accuracy = {'classifier': classifier, 'feature_name': feature_name,
                                'accuracy': accuracy, 'limit': limit,
                                'pos_precision': pos_precision, 'pos_recall': pos_recall,
                                'neg_precision': neg_precision, 'neg_recall': neg_recall,
                                'neu_precision': neu_precision, 'neu_recall': neu_recall}

    return classifier_with_accuracy

# a list of tubles of the different tokinator methods with its corresponding
# name and a boolean to indicate wether or not to parse a list of best words
# to filter in the tokinator process
methods = [('bag_of_words', tokinator.bag_of_words, False),
           ('bag of non stop words', tokinator.bag_of_non_stopwords, False),
           ('bag of best words', tokinator.bag_of_best_words, True),
           ('bag of bigrams words', tokinator.bag_of_bigrams_words, False),
           ('bag of best bigrams words', tokinator.bag_of_best_bigram_words, True)]

# a list to hold all the classification objects and their accuracy
method_outputs = []

# the outer loop calls different limits of best words to filter in the tokinator process
for limit in range(0, 10001, 100):
    print('classifying limit:', limit)

    # the inner loop calls different tokinator methods with the outer limit
    for method in methods:
        # the results are parsed to the outputs container
        if (limit > 0 and method[2]) or limit == 0:
            # Only evaluate the methods that don't use the limit once on limit == 0
            method_outputs.append(evaluate(method[0], method[1], method[2], limit))

# Sorts all the outputs by their accuracy and filters on the top 5.
method_outputs_top = sorted(method_outputs, key=lambda w_s: w_s['accuracy'], reverse=True)[:5]

# Prints out all the precision information on the top classification results
for method_output in method_outputs_top:
    print('\ntokinator:', method_output['feature_name'])
    print('best words limit:', method_output['limit'])
    print('accuracy:', method_output['accuracy'])
    print('pos precision:', method_output['pos_precision'])
    print('pos recall:', method_output['pos_recall'])
    print('neg precision:', method_output['neg_precision'])
    print('neg recall:', method_output['neg_recall'])
    print('neu precision:', method_output['neu_precision'])
    print('neu recall:', method_output['neu_recall'])
    method_output['classifier'].show_most_informative_features(5)

