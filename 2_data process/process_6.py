testlist=[]
fp=open(r'fce_train_NER.txt','r',encoding='UTF-8')
lines=fp.readlines()
for line in lines:
    line=line.strip().split(' ')
    if line =='':
        continue
    if len(line)>2:
        testlist.append(line[0]+' '+line[1]+'\n')
    else:
        testlist.append('\n')
fp.close()


with open('fce_train_NER_only.txt', "w", encoding='utf-8') as f:
    for i in range(0, len(testlist)):
        f.write(testlist[i])
