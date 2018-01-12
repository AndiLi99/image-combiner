import csv
import urllib

with open("../data/painting_dataset.csv") as f:
    reader = csv.reader(f)
    i=1
    for row in reader:
        i+=1
        string = row[2]
        f = open("../data/images/"+str(i).zfill(5))
