import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
news = open('./datas/Corpus_utf8.txt', 'r',encoding='utf8')
model = Word2Vec(LineSentence(news), sg=0,size=200, window=5, min_count=5, workers=12)
model.save("news.word2vec")

model = gensim.models.KeyedVectors.load("news.word2vec")

print(123)
