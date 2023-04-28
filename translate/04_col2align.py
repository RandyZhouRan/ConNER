with open('sample.txt') as srcf, open('unlabel.txt','w') as tgtf:
    for sent in srcf.read().rstrip().split('\n\n'):
        count = 0
        for line in sent.split('\n'):
            tok, label = line.split()
            if label[:2] in ['B-', 'S-']:
                count += 1
            if label == 'O':
                tgtf.write('\t'.join([tok, '0', label]) + '\n')
            else:
                tgtf.write('\t'.join([tok, str(count), label]) + '\n')
        tgtf.write('\n')

with open('en.trans.iobes.txt') as srcf, open('trans.txt', 'w') as tgtf:
    for sent in srcf.read().rstrip().split('\n\n'):
        for line in sent.split('\n'):
            tok, label_with_align = line.split()
            if label_with_align == 'O':
                tgtf.write('\t'.join([tok, '0', label_with_align]) + '\n')
            else:
                label = ''
                align = ''
                for s in label_with_align:
                    if s.isdigit():
                        align += s
                    else:
                        label += s
                tgtf.write('\t'.join([tok, str(int(align)+1), label]) + '\n')
        tgtf.write('\n')