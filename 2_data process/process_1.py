testlist=[]
fp=open(r'fce_train.txt','r',encoding='UTF-8')
lines=fp.readlines()
for i in range(10000):
    testlist.append(lines[i])


fp.close()

with open('training_sample.txt', "w", encoding='utf-8') as f:
    for i in range(0, len(testlist)):
        f.write(testlist[i])
