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
opos=[]

tem=[]
tlabel=[]
tpos=[]


for line in sentences:
    for i in line:
        tem.append(i[0])
        tlabel.append(i[1])
    pos=nltk.pos_tag(tem)
    for i in range(len(pos)):
        tpos.append(pos[i][1])
    output.append(tem)
    olabel.append(tlabel)
    opos.append(tpos)
    tem=[]
    tlabel=[]
    tpos=[]

with open('fce_train_POS.txt', "w", encoding='utf-8') as f:
    for i in range(len(output)):
        for i1 in range(len(output[i])):
            f.write(output[i][i1])
            f.write(' ')
            f.write(opos[i][i1])
            f.write(' ')
            f.write(olabel[i][i1])
            f.write('\n')
            if i1==len(output[i])-1:
                f.write("\n")