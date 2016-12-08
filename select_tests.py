import csv

in_file = 'data/all_text_actions_test.csv'
out_file = 'data/all_text_actions_test_select.csv'
select_file = 'data/selected.csv'

with open(in_file, 'r', encoding='utf8') as csv_input:
    with open(select_file, 'r', encoding='utf8') as csv_select:
        with open(out_file, 'w', encoding='utf8') as csv_output:
            reader = csv.reader(csv_input, delimiter=';')
            select = csv.reader(csv_select, delimiter=',')
            writer = csv.writer(csv_output, delimiter=';', lineterminator='\n')

            new_data = []
            row = next(reader)
            row.append('Test')
            new_data.append(row)
            match_count = 0
            for row in reader:
                text = row[21]
                for select_row in select:
                    text_select = select_row[0]

                    if text == text_select:
                        row.append('Yes')
                        row[23] = select_row[1]
                        match_count += 1
                        break

                new_data.append(row)

            print(match_count)
            writer.writerows(new_data)
