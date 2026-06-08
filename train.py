import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from feature_extractor import extract_features

data = pd.read_csv('dataset.csv')

X = data['email']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

X_train_text = vectorizer.fit_transform(X_train)
X_test_text = vectorizer.transform(X_test)

X_train_extra = extract_features(X_train)
X_test_extra = extract_features(X_test)

X_train_final = hstack([X_train_text, X_train_extra.values])
X_test_final = hstack([X_test_text, X_test_extra.values])

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

pred = model.predict(X_test_final)

print('Accuracy:', accuracy_score(y_test, pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, pred))

joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
