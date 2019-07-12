from collections import Counter
import tensorflow as tf 

class Poem(object):
    def __init__(self, filename):
        poems = []
        file = open(filename, 'r', encoding='UTF-8')
        for line in file:
            title, author, poem = line.strip().split("::")
            poem = poem.replace(' ', '')
            if len(poem) < 10 or len(poem) > 80:
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            # poem = '[' + poem + ']'
            if len(poem) < 80:
                poem = poem.ljust(80, " ")
            poems.append(poem)
        
        word_frequence = Counter()
        for poem in poems:
            word_frequence.update(poem)
        # print(word_frequence.items())

        word_frequence[" "] = -1 
        word_pairs = sorted(word_frequence.items(), key = lambda x: x[1], reverse=True)
        self.words, frequence = zip(*word_pairs)
        self.word_num = len(self.words)
        self.word_to_id = dict(zip(self.words, range(self.word_num)))
        self.id_to_word = dict(zip(range(self.word_num), self.words))
        self.poems_vector = [[self.word_to_id[word] for word in poem] for poem in poems]
        
        # print(poems_vector)
        # for poem in poems_vector:
        #     print(len(poem))

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.poems_vector)
        return dataset 
        

if __name__ == '__main__':
    poem = Poem('../dataset/poetrySong/poetrySongTest.txt')
    