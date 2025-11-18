import numpy as np

sentence = "This is is a sample sentence with several words"

dict1 = {}
words = sentence.split()
def word_count(sentence):
    words = sentence.split()
    for word in words:
        if word in dict1:
            dict1[word] += 1
        else:
            dict1[word] = 1

    return dict1

print(word_count(sentence))