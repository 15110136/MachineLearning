import csv
import math

Tt = []

data = {}


with open("data.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = ',', quotechar='"')
    for row in reader:
        if(row[0]!='STATION'):
            if not(row[2] in data):
                data[row[2]] = [float('nan'), float('nan'), float('nan'), float('nan')]
            if row[1] == "MUGLA, TU":
                data[row[2]][0] = float(row[3])
            if row[1] == "BALIKESIR, TU":
                data[row[2]][1] = float(row[3])
            if row[1] == "AYDIN, TU":
                data[row[2]][2] = float(row[3])
            if row[1] == "IZMIR GUZELYALI, TU":
                data[row[2]][3] = float(row[3])

sortedData = sorted(data.items(), key = lambda kv: kv[0])

sortedData = sortedData[8430:]

for row in range(len(sortedData)):
    for i in range(4):
        if math.isnan(sortedData[row][1][i]):
            sortedData[row][1][i] = round(sortedData[row-1][1][i]*0.5 + sortedData[row-1][1][i]*0.25 + sortedData[row-1][1][i]*0.15 + sortedData[row-1][1][i]*0.1, 1)

T = [row[1] for row in sortedData]

#t = n0 + n1*t1 + n2*t2 + n3*t3 + n4*t4 + n5*t5 + n6*t6

n = [0.75, 1, 1, 1, 1, 1, 1]

N = len(T)
Nt = int(N/10)
N -= Nt

T = [[x[0]/25, x[1]/25, x[2]/25, x[3]/25] for x in T]

alpha = 0.0001

def calcMSE():
    ae = 0
    for i in range(5, N):
        ae += ((T[i][3] - (n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2]))**2)/N
    return math.sqrt(ae)

def calDer(x):
    res = 0
    if x == 0:
        for i in range(5,N):
            res += -(T[i][3] - (n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2]))
    if x <= 3 and x >= 1:
        for i in range(5,N):
            res += -T[i-x][3]*(T[i][3]-(n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2]))
    if x > 3:
        for i in range(5,N):
            res += -T[i-1][x-4]*(T[i][3]-(n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2]))
    return res;

def calError():
    ave = 0
    for i in range(N,N+Nt):
        ave += abs((T[i][3] - (n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2])))/Nt
        print(
            "%.2f %.3f" % (T[i][3], (n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2])),
            T[i][3] - (n[0] + n[1]*T[i-1][3] + n[2]*T[i-2][3] + n[3]*T[i-3][3] + n[4]*T[i-1][0] + n[5]*T[i-1][1] + n[6]*T[i-1][2])
        )
    print(ave)
"""
while abs(calDer(1))>0.01:
    nn = [0, 0, 0, 0, 0, 0, 0]
    for j in range(7):
        nn[j] = n[j] - alpha*calDer(j)

    n = nn
    print(n, calcMSE())
"""

n = [1.3222737946806093, 1.014403495152182, -0.4245195209884346, 0.17905047145596262, 0.15448111463168568, 0.0373477100009339, -0.005824989667334112]
#n[0] *= 25
T = [[x[0]*25, x[1]*25, x[2]*25, x[3]*25] for x in T]

print(n)
calError()
