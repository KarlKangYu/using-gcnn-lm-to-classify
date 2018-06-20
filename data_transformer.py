import codecs
import sys
from data_utility import DataUtility
import os

def word_id(data, vocab_path):
    if not os.path.exists(vocab_path):
        os.mkdir(vocab_path)
    with codecs.open(data, "r") as f:
        with codecs.open(os.path.join(vocab_path, "vocab_in_words"), "w") as f1:
            with codecs.open(os.path.join(vocab_path, "vocab_out"), "w") as f2:
                count = 0
                for line in f.readlines():
                    word, _ = line.strip().split('\t')
                    word_lower = word.lower()
                    f1.write(word_lower + '##' + str(count) + '\n')
                    f2.write(word + '##' + str(count) + '\n')
                    count += 1

def letter_id(letter_data, vocab_path):
    if not os.path.exists(vocab_path):
        os.mkdir(vocab_path)
    with codecs.open(letter_data, "r") as f:
        with codecs.open(os.path.join(vocab_path, "vocab_in_letters"), "w") as f1:
            count = 0
            for line in f.readlines():
                letter = line.strip()
                if letter == '#':
                    letter = '<unk>'
                elif letter == '^':
                    letter = '<start>'
                elif letter == '$':
                    continue
                f1.write(letter + '##' + str(count) + '\n')
                count += 1

def train_in_ids_lm(train_data, vocab_path):
    if not os.path.exists(vocab_path):
        os.mkdir(vocab_path)
    vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")

    vocab_file_out = os.path.join(vocab_path, "vocab_out")


    data_ut = DataUtility(vocab_file_in_words=vocab_file_in_words,
                               vocab_file_out=vocab_file_out)


    with codecs.open(train_data, "r") as f:
        with codecs.open(os.path.join(vocab_path, "train_in_ids_lm"), "w") as f1:
            for line in f.readlines():
                words = line.strip()
                words = words.replace('.', ' .')
                words = words.replace(',', ' ,')
                words = words.split()
                words_ids = data_ut.words2ids(words)
                words_ids = [str(id) for id in words_ids]
                words_ids = ' '.join(words_ids)
                f1.write(words_ids + '#' + words_ids + '\n')

def train_in_ids_letters(train_data, vocab_path, emoji_data):
    if not os.path.exists(vocab_path):
        os.mkdir(vocab_path)
    vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
    vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
    vocab_file_out = os.path.join(vocab_path, "vocab_out")


    data_ut = DataUtility(vocab_file_in_words=vocab_file_in_words, vocab_file_in_letters=vocab_file_in_letters,
                               vocab_file_out=vocab_file_out)

    emojis = []
    with codecs.open(emoji_data, "r") as f:
        for line in f.readlines():
            emoji, _ = line.strip().split('\t')
            emojis.append(emoji)

    with codecs.open(train_data, "r") as f:
        with codecs.open(os.path.join(vocab_path, "train_in_ids_letters"), "w") as f1:
            for line in f.readlines():
                letters, _ = line.strip().split('\t')
                letters = letters.split('#')#letters = ['where', 'so', 'you', 'want', 'me', 'tk', 'ride', '?', '!', 'baby']
                letters_ids = ['1']
                for word in letters:#word = 'where'
                    if word in emojis:
                        letters_id = '1'
                    else:
                        letter = []
                        for i in word:
                            letter.append(i)#letter = ['w', 'h', 'e', 'r', 'e']
                        letter = ' '.join(letter)#letter = 'w h e r e'
                        letters_id = data_ut.letters2ids(letter)#letters_id = [1, 25, 10, 7, 20, 7]
                        letters_id = [str(id) for id in letters_id]#['1', '25', ..., '7']
                        letters_id = ' '.join(letters_id)#letters_id = '1 25 10 7 20 7'
                    letters_ids.append(letters_id)#letters_ids = ['1', '1', '1 25 10 7 20 7', ...,]
                f1.write('#'.join(letters_ids) + '\n')


if __name__ == '__main__':
    args = sys.argv
    vocab_path = args[1]
    # word_data_path = args[2]
    # letter_data_path = args[3]
    train_data = args[2]
    # emoji_data = args[5]

    # word_id(word_data_path, vocab_path)
    # letter_id(letter_data_path, vocab_path)
    train_in_ids_lm(train_data, vocab_path)
    # train_in_ids_letters(train_data, vocab_path, emoji_data)

