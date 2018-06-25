# https://stackoverflow.com/questions/2512386/how-to-merge-200-csv-files-in-python

allFiles = open("output.csv", "a")

for line in open("result1.csv"):
    allFiles.write(line)

# for the rest
for number in range(2, 12):
    file = open("result"+str(number)+".csv")
    file.next()
    for line in file:
        allFiles.write(line)
    file.close()

allFiles.close()

