import csv
with open('cleaned_data.csv', 'rb') as csvfile:
    mydata = csv.reader(csvfile, delimiter=';')
    for row in mydata:
        print(', '.join(row))
# with open('eggs.csv', 'w', newline='') as csvfile:
    # spamwriter = csv.writer(csvfile, delimiter=' ',
# quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
