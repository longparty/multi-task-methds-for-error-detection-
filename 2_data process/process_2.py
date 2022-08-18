output=[]
olabel=[]
tem=[]
tlabel=[]
fp=open(r'fce.train.gold.bea19.m2','r',encoding='UTF-8')
lines=fp.readlines()
count=-1
for line in lines:
    line=line.strip().split(' ')
    if line =='':
        continue
    if line[0]=="S":
        for i in range(1,len(line)):
            tem.append(line[i])
            tlabel.append("c")
        count+=1
        output.append(tem)
        olabel.append(tlabel)
        tem=[]
        tlabel=[]
    if line[0]=="A":
        x=int(line[1])
        # print(count)
        # print(x)
        if x==-1:
            continue
        else:
            if x>len(olabel[count])-1:
                continue
            else:
                olabel[count][x]='i'
    else:
        continue
fp.close()
with open('fce_train.txt', "w", encoding='utf-8') as f:
    for i in range(len(output)):
        for i1 in range(len(output[i])):
            f.write(output[i][i1])
            f.write(' ')
            f.write(olabel[i][i1])
            f.write('\n')
            if i1==len(output[i])-1:
                f.write("\n")

