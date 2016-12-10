import csv
import os
import language_detection


def file_len(file_name, encoding):
    with open(file_name, encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def add_language_to_csv(in_file_path, default_language, site_admin, test_limit, text_column=21, encoding='utf8', delim=';'):
    total_rows = file_len(in_file_path + '.csv', encoding)
    with open(in_file_path + '.csv', 'r', encoding=encoding) as csv_input:
        reader = csv.reader(csv_input, delimiter=delim)

        row = next(reader)
        if not new_data:
            row.append('Language')
            row.append('Sentiment')
            row.append('File')
            row.append('Admin')
            new_data.append(row)
            test_data.append(row)

        current_row = 1
        test_count = 0
        print('\nAdding language to ' + in_file_path)
        print('Total rows: ' + str(total_rows))

        for row in reader:
            if len(row[text_column]) > 0:
                language = language_detection.detect_language(row[text_column], default_language)
                row.append(language_detection.detect_language(row[text_column], default_language))
                row.append('')
                row.append(in_file_path)

                if row[10] == site_admin:
                    row.append('Yes')
                else:
                    row.append('No')

                    if language == 'english' and test_count < test_limit:
                        test_count += 1
                        test_data.append(row)

                new_data.append(row)

            if current_row % 1000 == 0:
                print('\x1b[2K\r' + str(round(current_row * 100 / total_rows)) + '%', end='')
            current_row += 1


def create_output_csv(in_files):

    outfile_path = 'data/all_text_actions.csv'
    testfile_path = 'data/all_text_actions_test.csv'
    try:
        os.remove(outfile_path)

    except FileNotFoundError:
        pass
    try:
        os.remove(testfile_path)

    except FileNotFoundError:
        pass

    with open(outfile_path, 'w', encoding='utf8') as csv_output:
        with open(testfile_path, 'w', encoding='utf8') as csv_test:
            writer = csv.writer(csv_output, delimiter=';', lineterminator='\n')
            test = csv.writer(csv_test, delimiter=';', lineterminator='\n')

            for in_file in in_files:
                add_language_to_csv('data/' + in_file[0], in_file[1], in_file[2], in_file[3])

            writer.writerows(new_data)
            test.writerows(test_data)

test_data = []
new_data = []

input_csv = [('FullRawFile_easyJet_66', 'english', 'easyJet', 145), ('FullRawFile_Norwegian_74', 'norwegian', 'Norwegian', 25),
             ('FullRawFile_Ryanair_94', 'english', 'Ryanair', 120), ('FullRawFile_SAS_73', 'swedish', 'SAS', 60),
             ('FullRawFile_AerLingus_64', 'english', 'Aer Lingus', 150), ('FullRawFile_Eurowings_19', 'english', 'Eurowings', 150),
             ('FullRawFile_Lufthansa_18', 'german', 'Lufthansa', 150)]

create_output_csv(input_csv)
