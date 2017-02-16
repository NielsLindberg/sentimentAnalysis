import collections
import nltk.classify.util
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import tokinator
import csv
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import time
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[\w']+")
time_str = time.strftime("%Y%m%d-%H%M%S")
training_path = 'data/all_text_actions_test_post_sod.csv'
training_path_with_comments = 'data/all_text_actions_test_sod.csv'
encoding = 'utf8'
delim = ';'


# file len is used to calculate the length of the file, which is used to calculate current percentages of
# various operations that loop through records in a file.
def file_len(file_name, encoding):
    with open(file_name, encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Word extend is used when we want to split the words from each input but add them all together in the same list and not
# one sub list per text record
def word_extend(data):
    data_new = []
    for text in data:
        word_filter = [i.lower() for i in tokenizer.tokenize(text)]
        data_new.extend(word_filter)
    return data_new


# Word split is used when we want to split the tokenz per input into different items in a list
def word_split(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in tokenizer.tokenize(word)]
        data_new.append(word_filter)
    return data_new


# Split sets takes a csv path, columns for text and sentiment aswell as bestwords limit
# as input, it returns lists for positive, neutral, negative and all records aswell as a list
# of the most important words for each label according to the limit
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

    bad_words = stopwords.words('english')

    observed_words_pos = word_extend(posdata)
    observed_words_no_bad_pos = [w for w in observed_words_pos if w not in bad_words]
    for word in observed_words_no_bad_pos:
        word_fd[word.lower()] += 1
        label_word_fd['Positive'][word.lower()] += 1

    observed_words_neg = word_extend(negdata)
    observed_words_no_bad_neg = [w for w in observed_words_neg if w not in bad_words]
    for word in observed_words_no_bad_neg:
        word_fd[word.lower()] += 1
        label_word_fd['Negative'][word.lower()] += 1

    observed_words_neu = word_extend(neudata)
    observed_words_no_bad_neu = [w for w in observed_words_neu if w not in bad_words]
    for word in observed_words_no_bad_neu:
        word_fd[word.lower()] += 1
        label_word_fd['Neutral'][word.lower()] += 1

    pos_word_count = label_word_fd['Positive'].N()
    neg_word_count = label_word_fd['Negative'].N()
    neu_word_count = label_word_fd['Neutral'].N()
    total_word_count = pos_word_count + neg_word_count + neu_word_count

    neg_scores = {}
    pos_scores = {}
    neu_scores = {}

    for word, freq in word_fd.items():
        pos_scores[word] = BigramAssocMeasures.chi_sq(label_word_fd['Positive'][word],
                                                      (freq, pos_word_count), total_word_count)
        neg_scores[word] = BigramAssocMeasures.chi_sq(label_word_fd['Negative'][word],
                                                      (freq, neg_word_count), total_word_count)
        neu_scores[word] = BigramAssocMeasures.chi_sq(label_word_fd['Neutral'][word],
                                                      (freq, neu_word_count), total_word_count)

    # applying the limit to each label as words in neutral and positive labels automatically
    # have higher scores due to most of the records being negative
    best_pos = sorted(pos_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:limit]
    best_neg = sorted(neg_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:limit]
    best_neu = sorted(neu_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:limit]
    best_words = set([w for w, s in best_pos + best_neg + best_neu])
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
    # call function read the text(21) and split into sentiment(23) return a best words by limit
    neudata, posdata, negdata, alldata, best_words = split_sets(training_path_with_comments, 21, 23, limit)

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
                                'accuracy': accuracy, 'limit': limit, 'best_words_list': best_words,
                                'pos_precision': pos_precision, 'pos_recall': pos_recall,
                                'neg_precision': neg_precision, 'neg_recall': neg_recall,
                                'neu_precision': neu_precision, 'neu_recall': neu_recall}

    return classifier_with_accuracy


def check_mutato_accuracy(training_path):
    with open(training_path, 'r', encoding=encoding) as csv_train:
        csv_reader = csv.reader(csv_train, delimiter=delim)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        results = []
        gold = []
        label = ''
        for i, row in enumerate(csv_reader):
            if row[23] == '2':
                label = 'Negative'
            elif row[23] == '1':
                label = 'Positive'
            elif row[23] == '0':
                label = 'Neutral'
            gold.append(label)
            refsets[label].add(i)
            testsets[row[26]].add(i)
            results.append(row[26])

        # Since we dont have a classifier object we manually calculate the accuracy
        # We do it the same way as nltk.util.accuracy but instead of using the classifier to compare
        # We just compare between the result and the gold standard (refset)
        correct = [l == r for (l, r) in zip(gold, results)]
        accuracy_manual = sum(correct) / len(correct)
        pos_precision = precision(refsets['Positive'], testsets['Positive'])
        pos_recall = recall(refsets['Positive'], testsets['Positive'])
        neg_precision = precision(refsets['Negative'], testsets['Negative'])
        neg_recall = recall(refsets['Negative'], testsets['Negative'])
        neu_precision = precision(refsets['Neutral'], testsets['Neutral'])
        neu_recall = recall(refsets['Neutral'], testsets['Neutral'])

        classifier_with_accuracy = {'classifier': 'N/A', 'feature_name': 'MUTATO',
                                    'feature_detector': 'MUTATO', 'best_words': False,
                                    'accuracy': accuracy_manual, 'limit': 0,
                                    'pos_precision': pos_precision, 'pos_recall': pos_recall,
                                    'neg_precision': neg_precision, 'neg_recall': neg_recall,
                                    'neu_precision': neu_precision, 'neu_recall': neu_recall}
        return classifier_with_accuracy


def classify_all(classifier_plus):
    with open(in_file_path, 'r', encoding=encoding) as csv_input:
        with open(out_file_path, 'w', encoding=encoding) as csv_output:
            reader = csv.reader(csv_input, delimiter=delim)
            writer = csv.writer(csv_output, delimiter=';', lineterminator='\n')
            record_types = ['POST', 'POSTREPLY', 'COMMENT', 'COMMENTREPLY']
            new_data = []
            current_row = 0
            for row in reader:
                if row[22] == 'english': # and row[0] in record_types:
                    words = [i.lower() for i in tokenizer.tokenize(row[21])]
                    if classifier_plus['best_words']:
                        features = classifier_plus['feature_detector'](classifier_plus['best_words_list'], words)
                    else:
                        features = classifier_plus['feature_detector'](words)
                    row[23] = classifier_plus['classifier'].classify(features)
                new_data.append(row)
                if current_row % 1000 == 0:
                    print('\x1b[2K\r Classifying: ' + str(round(current_row * 100 / total_rows)) + '%', end='')
                current_row += 1
            writer.writerows(new_data)


# a list of tubles of the different tokinator methods with its corresponding
# name and a boolean to indicate wether or not to parse a list of best words
# to filter in the tokinator process
methods = [('bag of non stop words', tokinator.bag_of_non_stopwords, False),
           ('bag of best words', tokinator.bag_of_best_words_non_stopwords, True),
           ('bag of bigrams words', tokinator.bag_of_bigram_words, False),
           ('bag of best bigrams words', tokinator.bag_of_best_bigram_words, True)]

# a list to hold all the classification objects and their accuracy
method_outputs = []

# the outer loop calls different limits of best words to filter in the tokinator process
for limit in range(500, 2001, 500):
    print('classifying limit:', limit)

    # the inner loop calls different tokinator methods with the outer limit
    for method in methods:
        # the results are parsed to the outputs container
        if (limit > 0 and method[2]) or limit == 0:
            # Only evaluate the methods that don't use the limit once on limit == 0
            method_outputs.append(train(method[0], method[1], method[2], limit))

# Sorts all the outputs by their accuracy and filters on the top 5.
method_outputs_top = sorted(method_outputs, key=lambda w_s: w_s['accuracy'], reverse=True)[:5]

method_outputs_top.append(check_mutato_accuracy(training_path_with_comments))
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

classify_all(best_method)
