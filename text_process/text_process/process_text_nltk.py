from nltk.stem.porter import PorterStemmer
#using NLTK to stem and lemma
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

class TextFeatureExtractor:

    def strip_stop_words(self, words):
    
        words = [w for w in words if w not in stopwords.words("english")]
        return words

    def stem(self, words):
    
        stemmed = [PorterStemmer().stem(w) for w in words]
        return stemmed

    def lemmatize(self, words):

        lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
        return lemmed

    def tokenize(self, text):
        words = word_tokenize(text)
        return words




