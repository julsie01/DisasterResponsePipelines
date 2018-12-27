from .process_text_nltk import TextFeatureExtractor
import re



def tokenize(text):

    """Input: text
        Output: text as tokens

        Function to tokenize text, strip stop words, stem and lemmatize
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    extractor = TextFeatureExtractor()
    words  = extractor.tokenize(text)
    tokens = extractor.strip_stop_words(words)
    tokens = extractor.stem(words)
    tokens = extractor.lemmatize(words)
    
    return tokens

if __name__ == '__main__':
    main()