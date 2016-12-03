import csv
import os
import language_detection


def file_len(file_name, encoding):
    with open(file_name, encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def add_language_to_csv(in_file_path, default_langauge, text_column=21, encoding='latin', delim=';'):
    total_rows = file_len(in_file_path + '.csv', encoding)
    with open(in_file_path + '.csv', 'r', encoding=encoding) as csv_input:
        with open(in_file_path + '_lang.csv', 'w', encoding=encoding) as csv_output:
            writer = csv.writer(csv_output, delimiter=delim, lineterminator='\n')
            reader = csv.reader(csv_input, delimiter=delim)

            new_data = []
            row = next(reader)
            row.append('Language')
            new_data.append(row)
            current_row = 1
            print('\nAdding language to ' + in_file_path)
            print('Total rows: ' + str(total_rows))

            for row in reader:
                if len(row[text_column]) > 0:
                    row.append(language_detection.detect_language(row[text_column], default_langauge))

                else:
                    row.append('Not text')
                new_data.append(row)
                if current_row % 1000 == 0:
                    print('\x1b[2K\r' + str(round(current_row * 100 / total_rows)) + '%', end='')
                current_row += 1

            writer.writerows(new_data)

# in_files = [('FullRawFile_easyJet_66', 'english'), ('FullRawFile_Norwegian_74', 'norwegian'),
#            ('FullRawFile_Ryanair_94', 'english'), ('FullRawFile_SAS_73', 'swedish'),
#            ('FullRawFile_AerLingus_64', 'english')]

in_files = [('FullRawFile_Eurowings_19', 'english'), ('FullRawFile_Norwegian_96', 'norwegian'), ('FullRawFile_Lufthansa_18', 'english')]

for in_file in in_files:
    try:
        os.remove('data/' + in_file[0] + '_lang.csv')

    except FileNotFoundError:
        pass

    add_language_to_csv('data/' + in_file[0], in_file[1])
