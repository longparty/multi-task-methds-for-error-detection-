import nltk

def read_input_files(file_paths):
    sentences = []
    line_length = None
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r", encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                sentences.append(sentence)
    return sentences

sentences = read_input_files('fce_train.txt')
output=[]
olabel=[]
oner=[]

tem=[]
tlabel=[]
tner=[]


for line in sentences:
    for i in line:
        tem.append(i[0])
        tlabel.append(i[1])
    pos=nltk.pos_tag(tem)
    ners = nltk.ne_chunk(pos)
    for i in range(len(ners)):
        if len(ners[i])==2 and isinstance(ners[i],tuple):
            tner.append('O')
        else:
            for i2 in range(len(ners[i].leaves())):
                tner.append(ners[i].label())
    output.append(tem)
    olabel.append(tlabel)
    oner.append(tner)
    tem=[]
    tlabel=[]
    tner=[]

with open('fce_train_NER.txt', "w", encoding='utf-8') as f:
    for i in range(len(output)):
        for i1 in range(len(output[i])):
            f.write(output[i][i1])
            f.write(' ')
            f.write(oner[i][i1])
            f.write(' ')
            f.write(olabel[i][i1])
            f.write('\n')
            if i1==len(output[i])-1:
                f.write("\n")