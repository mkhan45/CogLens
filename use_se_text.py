from gensim.models.keyedvectors import KeyedVectors as KV
import se_text as st

#Load files
with open("/Users/crystal/repositories/Student_Week3/bag_of_words/dat/stopwords.txt", 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

glove50 = KV.load_word2vec_format('/Users/crystal/Desktop/python-workspace/CogWorks2019/glove.6B.50d.txt.w2v', binary=False)

#Make an embedding
all_captions = ["cat", "dog", "bird eating", "mouse", "mouse flying"] #put the list of captions here
counters = st.to_counters(all_captions)
vocab = st.to_vocab(counters, stop_words=stops)
idf = st.to_idf(counters, vocab)
embedding = st.se_text("mouse flying", glove50, idf, vocab) #change "mouse flying" to whatever caption

print(embedding) #you can delete this if you want