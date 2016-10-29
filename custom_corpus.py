from nltk.corpus.reader import CategorizedPlaintextCorpusReader
import os

current_dir = os.getcwd()
os.chdir('D:/nltk_data/corpora/moviez')

reader = CategorizedPlaintextCorpusReader('.', r'.*\.txt',
                                          cat_pattern=r'(\w+)/*')
print(len(reader.fileids()))
print(reader.categories())
