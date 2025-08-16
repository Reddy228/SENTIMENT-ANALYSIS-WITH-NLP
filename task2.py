# Import libraries
import nltk
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK datasets
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load movie reviews data
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Create DataFrame
df = pd.DataFrame(documents, columns=['text', 'sentiment'])
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply text cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

# Train Logistic Regression Model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Accuracy & Report
print(f"\n✅ Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, predictions, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('📌 Confusion Matrix')
plt.tight_layout()
plt.show()
