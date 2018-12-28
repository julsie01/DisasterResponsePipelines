from nltk.stem.porter import PorterStemmer
#using NLTK to stem and lemma
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

class TextFeatureExtractor:


    def strip_stop_words(self, words):
        """
        Funcion to remove stop words from a list of words
        Input: List of Words
        Output: List of Words excluding nltk stopwords
        """
        words = [w for w in words if w not in stopwords.words("english")]
        return words

    def stem(self, words):
        """
        Funcion to reduce words to their base word (stemming)
        Input: List of Words
        Output: List of Words Stemmed using NLTK porter stemmer
        """
        stemmed = [PorterStemmer().stem(w) for w in words]
        return stemmed

    def lemmatize(self, words):
        """
        Funcion to lematize a list of words
        Input: List of Words
        Output: List of Words Lemmatized using NLTK Word Net Lemmatizer
        """
        lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
        return lemmed

    def tokenize(self, text):
        """
        Funcion to convert a text string to a list of individual words
        Input: text: String of Text
        Output: List of individual Words
        """
        words = word_tokenize(text)
        return words




