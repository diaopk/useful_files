from __future__ import division
from mrjob.job import MRJob
from mrjob.step import MRStep
import os
import math
import re

WORD_RE = re.compile(r"[\w']+")
cwd = os.getcwd()

# method to calculate the TF for a word in a document
# return the input and its tf value as tuple
def get_tf(word, doc):
    num_word = 0
    total = 0
    for w in WORD_RE.findall(doc):
        total += 1
        if w == word:
            num_word += 1
    
    return num_word/total

# method to calculate the idf
def get_idf(total_doc_num, doc_num_with_word):
    return math.log10(total_doc_num/doc_num_with_word)

class MRWordFrequencyCount(MRJob):

    # init data structure
    # nest dict defined
    def mapper_init(self):
        # tf_dict = {user_id: {word1: tf_value, word2: tf_value}}
        self.tf_dict = {}
        # idf_dict = {user_id: [total_doc_num, {doc_num_with_word, idf_value_word}]}
        self.idf_dict = {}
        self.tfidf_dict = {}

    def mapper_middle(self, _, line):
        #print('mapper_middle starts...')
        # split the user_id and body from source text file
        user_id, body = line.split(',')
        # set default dicts
        self.tf_dict.setdefault(user_id, {})
        self.idf_dict.setdefault(user_id, [0, {}])
        self.tfidf_dict.setdefault(user_id, {})
        
        # increment the total_doc_num for a user
        self.idf_dict[user_id][0] += 1

        # local vars
        # used to check if the same word in a doc(body)
        same_word_count = {}

	for word in WORD_RE.findall(body):
            # init same_word_count
            same_word_count.setdefault(word, 0)
            same_word_count[word] += 1

            # set default for tfidf_dict
            self.tfidf_dict[user_id].setdefault(word, 0)

            # set default for tf_dict
            self.tf_dict[user_id].setdefault(word, 0) 
            # replace it with larger tf value if there are duplicate words
            self.tf_dict[user_id][word] = max(self.tf_dict[user_id][word], get_tf(word, body))

            # set values for idf_dict
            if self.idf_dict[user_id][1].has_key(word) == False:
                # if a word hasn't set the doc_num_with_word
                # set [doc_num_with_word, idf_value_word] = [0, 0]
                self.idf_dict[user_id][1].setdefault(word, [1, 0])
            elif WORD_RE.findall(body).count(word) == 1:
                # elif there is only one word in body
                self.idf_dict[user_id][1][word][0] += 1
            else:
                # else case: word has been inited and there are more than one words
                # in the doc.
                if same_word_count[word] == WORD_RE.findall(body).count(word):
                    # if last same word is count
                    self.idf_dict[user_id][1][word][0] += 1
                else:
                    pass

        print('mapper_middle finishes....')

    def mapper_final(self):
        print('mapper_finall starts....')

        # calculate the idf values
        for user_id, user_dict in self.idf_dict.items():
            total_doc_num = user_dict[0]
            # loop through the words
            # word - str; word_values - doc_num_with_word, idf_value
            for word, word_values in user_dict[1].items():
                doc_num_with_word = word_values[0]
                # calculate the idf value and update idf_dict
                idf_val = get_idf(total_doc_num, doc_num_with_word)
                self.idf_dict[user_id][1][word][1] = idf_val
                tf_val = self.tf_dict[user_id][word]
                # calculate the tf-idf values
                tfidf = tf_val * idf_val
                
                yield (user_id+'-'+word, tfidf)
    
    def combiner(self, user_id, val):
	yield (user_id, sum(val))

    def reducer(self, key, values):
        yield (key, sum(values))

    def steps(self):
        return [MRStep(mapper_init=self.mapper_init,
                        mapper=self.mapper_middle,
                        mapper_final=self.mapper_final,
                        combiner=self.combiner,
                        reducer=self.reducer
                )]

if __name__ == '__main__':
    MRWordFrequencyCount.run()
