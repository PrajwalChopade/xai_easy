from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from xai_easy import explain_model, explain_instance, save_html_report
import pandas as pd

texts = ["good product", "bad service", "not recommended", "excellent", "horrible", "average", "will buy again", "never buying"]
labels = [1,0,0,1,0,1,1,0]
vec = CountVectorizer(ngram_range=(1,2), max_features=2000)
X = vec.fit_transform(texts).toarray()
X = pd.DataFrame(X, columns=vec.get_feature_names_out())
y = labels
clf = MultinomialNB().fit(X, y)
gdf = explain_model(clf, X, y, top_n=10)
print(gdf)
ldf = explain_instance(clf, X, X.iloc[0].values)
save_html_report(gdf, ldf, title="Sentiment Demo", filename="sentiment_report.html")
print("Saved sentiment_report.html")
