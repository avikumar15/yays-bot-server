# Using Naiva Bayes for classification

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Loading Data

count_vect = CountVectorizer()
X_bot_counts = count_vect.fit_transform(chatbot_input.data)
X_bot_counts.shape

# TF-IDF

tfidf_transformer = TfidfTransformer()
X_bot_tfidf = tfidf_transformer.fit_transform(X_bot_counts)
X_bot_tfidf.shape


# Applying Naive Bayes algorithm 

clf = MultinomialNB().fit(X_bot_tfidf, chat_bot.target)

# Building pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(chat_bot.data, chat_bot.target)

# Predicting
predicted = text_clf.predict(chat_bot_test.data)

return predicted
