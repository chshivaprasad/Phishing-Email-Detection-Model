import re
import pandas as pd

KEYWORDS = ['verify','urgent','click','password','bank','login','claim','prize']

def extract_features(texts):
    rows = []
    for text in texts:
        text = str(text).lower()
        urls = len(re.findall(r'http[s]?://\S+', text))
        keywords = sum(k in text for k in KEYWORDS)
        rows.append([urls, keywords])
    return pd.DataFrame(rows, columns=['url_count','keyword_count'])
