output=[]
olabel=[]
oclass=[]

tem=[]
tlabel=[]
tclass=[]
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
            tclass.append("noop")
        count+=1
        output.append(tem)
        olabel.append(tlabel)
        oclass.append(tclass)
        tem=[]
        tlabel=[]
        tclass=[]
    if line[0]=="A":
        x=int(line[1])
        classlist=line[2].strip().split('|||')
        if x==-1:
            continue
        else:
            if x>len(olabel[count])-1:
                continue
            else:
                olabel[count][x]='i'
                oclass[count][x] = classlist[1]
    else:
        continue
fp.close()
with open('fce_train_error.txt', "w", encoding='utf-8') as f:
    for i in range(len(output)):
        for i1 in range(len(output[i])):
            f.write(output[i][i1])
            f.write(' ')
            f.write(oclass[i][i1])
            f.write(' ')
            f.write(olabel[i][i1])
            f.write('\n')
            if i1==len(output[i])-1:
                f.write("\n")

