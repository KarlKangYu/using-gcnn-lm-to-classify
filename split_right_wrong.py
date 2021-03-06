import codecs
import sys

def split(data_dir, pos_dir, neg_dir):
    with codecs.open(data_dir, 'r') as f1:
        with codecs.open(pos_dir, 'w') as f2:
            with codecs.open(neg_dir, 'w') as f3:
                for line in f1.readlines():
                    line = line.strip()
                    neg, pos = line.split('\t')
                    neg = neg[:-1]
                    pos = pos[:-1]
                    f2.write(pos + '\n')
                    f3.write(neg + '\n')

if __name__ == "__main__":
    args = sys.argv
    data_dir = args[1]
    pos_dir = args[2]
    neg_dir = args[3]
    split(data_dir, pos_dir, neg_dir)



