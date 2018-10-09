import csv

data = []

file = "temp-weat.csv"

with open(file) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',', quotechar='"')
    for row in reader:
        data.append(row)

N = len(data)
T = int(N/10)

test = data[:T]
data = data[T:]

guess = []

correct = 0
wrong = 0

for i in range(5, len(test)):
    neig = []
    for j in range(5, len(data)):
        dist = 0
        for k in range(1, 5):
            dist += (abs(float(test[i-k][1]) - float(data[j-k][1])) + int(test[i-k][2] != data[j-k][2])*20  )
#            print(data[j-k][2], test[i-k][2], int(data[j-k][2]!=test[i-k][2])*20)
        neig.append((dist, data[j][2]))
    neig=sorted(neig)[:10]
    dict = {}
    for j in range(10):
        if not(neig[j][1] in dict):
            dict[neig[j][1]] = 10-j
        else:
            dict[neig[j][1]] += 10-j

    res = sorted(dict.items(), key = lambda kv: kv[1])
    res.reverse()
    print(test[i][2], res[0][0])
    if(test[i][2] == res[0][0]):
        correct += 1
    if(test[i][2] != res[0][0]):
        wrong += 1

print(correct, wrong)
