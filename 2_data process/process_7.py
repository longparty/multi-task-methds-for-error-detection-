testlist=[]
fp=open(r'fce_train_NER.txt','r',encoding='UTF-8')
lines=fp.readlines()
for line in lines:
    line=line.strip().split(' ')
    if line =='':
        continue
    if len(line)>2:
        testlist.append(line[0]+' '+line[1]+' ')
    else:
        testlist.append('')
fp.close()

testlist2=[]
fp=open(r'fce_train_POS.txt','r',encoding='UTF-8')
lines=fp.readlines()
for line in lines:
    line=line.strip().split(' ')
    if line =='':
        continue
    if len(line)>2:
        testlist2.append(line[1]+' ')
    else:
        testlist2.append('')
fp.close()

testlist3=[]
fp=open(r'fce_train_error.txt','r',encoding='UTF-8')
lines=fp.readlines()
for line in lines:
    line=line.strip().split(' ')
    if line =='':
        continue
    if len(line)>2:
        testlist3.append(line[1]+' '+line[2]+'\n')
    else:
        testlist3.append('\n')
fp.close()

with open('fce_train_all.txt', "w", encoding='utf-8') as f:
    for i in range(0, len(testlist)):
        f.write(testlist[i])
        f.write(testlist2[i])
        f.write(testlist3[i])