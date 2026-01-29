import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# load saved model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# take input
news = input("Enter news text: ")

# clean & convert
cleaned = clean_text(news)
vector = vectorizer.transform([cleaned])

# predict
result = model.predict(vector)

if result[0] == 1:
    print("🟢 REAL NEWS")
else:
    print("🔴 FAKE NEWS")
