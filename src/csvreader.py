import csv
import urllib

with open("../data/painting_dataset.csv") as f:
    reader = csv.reader(f)
    i=1
    reader.next()
    for row in reader:
        i+=1
        string = row[2][1:-1]
        urllib.urlretrieve(string, "../data/painting_images/" + str(i).zfill(5) + ".jpg")
