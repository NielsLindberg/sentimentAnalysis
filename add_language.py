import csv
import os
import language_detection


def file_len(file_name, encoding):
    with open(file_name, encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def add_language_to_csv(in_file_path, text_column=21, encoding='latin', delim=';'):
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
            print('Adding language to ' + in_file_path)
            print('Total rows: ' + str(total_rows))

            for row in reader:
                if len(row[text_column]) > 10:
                    row.append(language_detection.detect_language(row[text_column]))

                else:
                    row.append('Not text')
                new_data.append(row)
                if current_row % 1000 == 0:
                    print('\x1b[2K\r' + str(round(current_row * 100 / total_rows)) + '%', end='')
                current_row += 1

            writer.writerows(new_data)

in_files = ['data/FullRawFile_airberlin_71']

for in_file in in_files:
    try:
        os.remove(in_file + '_lang.csv')

    except FileNotFoundError:
        pass

    add_language_to_csv(in_file)
