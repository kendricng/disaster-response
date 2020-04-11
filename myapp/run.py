from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# load function for joblib
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

from myapp import app

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    
if __name__ == '__main__':
    main()
