import collections
import nltk.classify.util
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
import tokinator
import csv
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import time


time_str = time.strftime("%Y%m%d-%H%M%S")
encoding = 'utf8'
delim = ';'


def file_len(file_name, encoding):
    with open(file_name, encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


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
        label_word_fd['Positive'][word.lower()] += 1

    for word in word_extend(negdata):
        word_fd[word.lower()] += 1
        label_word_fd['Negative'][word.lower()] += 1

    for word in word_extend(neudata):
        word_fd[word.lower()] += 1
        label_word_fd['Neutral'][word.lower()] += 1

    pos_word_count = label_word_fd['Positive'].N()
    neg_word_count = label_word_fd['Negative'].N()
    neu_word_count = label_word_fd['Neutral'].N()
    total_word_count = pos_word_count + neg_word_count + neu_word_count

    word_scores = {}

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['Positive'][word],
                                               (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['Negative'][word],
                                               (freq, neg_word_count), total_word_count)
        neu_score = BigramAssocMeasures.chi_sq(label_word_fd['Neutral'][word],
                                               (freq, neu_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score + neu_score

    best = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:limit]
    best_words = set([w for w, s in best])
    return neudata, posdata, negdata, alldata, best_words


def split_feats(label, data, feature_detector, use_best_words, best_words):
    if use_best_words:
        feats = [(feature_detector(best_words, f), label) for f in word_split(data)]

    else:
        feats = [(feature_detector(f), label) for f in word_split(data)]
    return feats


def create_train_test_sets(negfeats, posfeats, neufeats):
    negcutoff = int(len(negfeats) * 3 / 4)
    poscutoff = int(len(posfeats) * 3 / 4)
    neucutoff = int(len(neufeats) * 3 / 4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + neufeats[:neucutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + neufeats[neucutoff:]
    return trainfeats, testfeats


def train(feature_name, feature_detector, use_best_words, limit):

    # input csv file with manually classified posts
    data_path = 'D:/Workspace/CBS/BigSocialData/SentimentAnalysis/data/all_text_actions_test.csv'

    # call function read the text(21) and split into sentiment(23) return a best words by limit
    neudata, posdata, negdata, alldata, best_words = split_sets(data_path, 21, 23, limit)

    # only some of the tokinator help function uses the best_words input which is handled by the
    # use_best_words parameter
    negfeats = split_feats('Negative', negdata, feature_detector, use_best_words, best_words)
    posfeats = split_feats('Positive', posdata, feature_detector, use_best_words, best_words)
    neufeats = split_feats('Neutral', neudata, feature_detector, use_best_words, best_words)

    trainfeats, testfeats = create_train_test_sets(negfeats, posfeats, neufeats)

    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    accuracy = nltk.classify.util.accuracy(classifier, testfeats)
    pos_precision = precision(refsets['Positive'], testsets['Positive'])
    pos_recall = recall(refsets['Positive'], testsets['Positive'])
    neg_precision = precision(refsets['Negative'], testsets['Negative'])
    neg_recall = recall(refsets['Negative'], testsets['Negative'])
    neu_precision = precision(refsets['Neutral'], testsets['Neutral'])
    neu_recall = recall(refsets['Neutral'], testsets['Neutral'])

    classifier_with_accuracy = {'classifier': classifier, 'feature_name': feature_name,
                                'feature_detector': feature_detector, 'best_words': use_best_words,
                                'accuracy': accuracy, 'limit': limit,
                                'pos_precision': pos_precision, 'pos_recall': pos_recall,
                                'neg_precision': neg_precision, 'neg_recall': neg_recall,
                                'neu_precision': neu_precision, 'neu_recall': neu_recall}

    return classifier_with_accuracy

# a list of tubles of the different tokinator methods with its corresponding
# name and a boolean to indicate wether or not to parse a list of best words
# to filter in the tokinator process
methods = [('bag_of_words', tokinator.bag_of_words, False),
           ('bagof non stop words', tokinator.bag_of_non_stopwords, False),
           ('bag of best words', tokinator.bag_of_best_words, True),
           ('bag of bigrams words', tokinator.bag_of_bigram_words, False),
           ('bag of best bigrams words', tokinator.bag_of_best_bigram_words, True)]

# a list to hold all the classification objects and their accuracy
method_outputs = []

# the outer loop calls different limits of best words to filter in the tokinator process
for limit in range(0, 1001, 1000):
    print('classifying limit:', limit)

    # the inner loop calls different tokinator methods with the outer limit
    for method in methods:
        # the results are parsed to the outputs container
        if (limit > 0 and method[2]) or limit == 0:
            # Only evaluate the methods that don't use the limit once on limit == 0
            method_outputs.append(train(method[0], method[1], method[2], limit))

# Sorts all the outputs by their accuracy and filters on the top 5.
method_outputs_top = sorted(method_outputs, key=lambda w_s: w_s['accuracy'], reverse=True)[:5]


def check_sodata_accuracy(training_path):
    with open(training_path, 'r', encoding=encoding) as csv_train:
        csv_reader = csv.reader(csv_train, delimiter=delim)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, row in enumerate(csv_reader):
            refsets[row[23]].add(i)
            testsets[row[26]].add(i)

        pos_precision = precision(refsets['1'], testsets['Positive'])
        pos_recall = recall(refsets['1'], testsets['Positive'])
        neg_precision = precision(refsets['2'], testsets['Negative'])
        neg_recall = recall(refsets['2'], testsets['Negative'])
        neu_precision = precision(refsets['0'], testsets['Neutral'])
        neu_recall = recall(refsets['0'], testsets['Neutral'])

        classifier_with_accuracy = {'classifier': 'N/A', 'feature_name': 'SODATA',
                                    'feature_detector': 'SODATA', 'best_words': False,
                                    'accuracy': 'N/A', 'limit': 0,
                                    'pos_precision': pos_precision, 'pos_recall': pos_recall,
                                    'neg_precision': neg_precision, 'neg_recall': neg_recall,
                                    'neu_precision': neu_precision, 'neu_recall': neu_recall}
        return classifier_with_accuracy

# Add SODATA
training_path = 'data/all_text_actions_test.csv'
method_outputs_top.append(check_sodata_accuracy(training_path))


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
    # method_output['classifier'].show_most_informative_features(5)

# classify all
in_file_path = 'data/all_text_actions.csv'
out_file_path = 'data/all_text_actions_sentiment' + time_str + '.csv'

best_method = method_outputs_top[0]
total_rows = file_len(in_file_path, encoding)

with open(in_file_path, 'r', encoding=encoding) as csv_input:
    with open(out_file_path, 'w', encoding=encoding) as csv_output:
        reader = csv.reader(csv_input, delimiter=delim)
        writer = csv.writer(csv_output, delimiter=';', lineterminator='\n')

        new_data = []
        current_row = 0
        for row in reader:
            if row[22] == 'english':
                features = tokinator.bag_of_bigram_words(row[21])
                row[23] = best_method['classifier'].classify(features)
            new_data.append(row)
            if current_row % 1000 == 0:
                print('\x1b[2K\r Classifying: ' + str(round(current_row * 100 / total_rows)) + '%', end='')
            current_row += 1
        writer.writerows(new_data)



