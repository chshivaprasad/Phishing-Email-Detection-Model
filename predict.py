import joblib
from scipy.sparse import hstack
from feature_extractor import extract_features

model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

email = input('Enter email text: ')

text_features = vectorizer.transform([email])
extra_features = extract_features([email])

final_features = hstack([text_features, extra_features.values])

prediction = model.predict(final_features)[0]
print('Prediction:', prediction)
