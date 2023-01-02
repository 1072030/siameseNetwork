import json

with open('report.json') as malData:
    malData = json.load(malData)
started = malData['info']['started']
ended = malData['info']['ended']
time = ended - started
timeUnit = time / 16
count = 0
block = {}
for i in range(16):
    block[i] = []
for i in malData['behavior']['processes']:
    for j in i['calls']:
        for k in range(16):
            if(started + timeUnit * (k + 1) > j['time'] >= started + timeUnit * k):
                block[k].append(j['time'])
                count += 1
print(block)
print(block.keys())
print(count)
