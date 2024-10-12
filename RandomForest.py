import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Sample Data: Questions and their corresponding hashtags
questions = [
    "Did you renew your insurance this year?",
    "Did you raise complain about your lost vehicle?",
    "Was your business affected due to theft?",
    "Did you file a claim for personal loss?",
    "Do you own multiple insurance policies?",
    "Has there been any suspicious activity in your business account?",
    "Did you experience motor theft last year?",
    "Was there a personal loss in the last 6 months?",
    "Has your business filed a claim for theft?",
    "Have you ever reported an insurance fraud?",
    "Did you claim for a vehicle-related accident?",
    "Was your personal vehicle stolen?",
    "Did your business face any insurance issues?",
    "Did you file a claim for a stolen vehicle?",
    "Has your personal account faced any fraudulent transactions?",
    "Has your insurance been renewed within the last year?",
    "Did you raise a claim for property damage?",
    "Was your vehicle damaged in a theft?",
    "Did your business report a fraud incident?",
    "Do you have any active insurance policies?"
]

hashtags = [
    ['#claims', '#insurance'],
    ['#motortheft', '#personalloss'],
    ['#business', '#motortheft'],
    ['#claims', '#personalloss'],
    ['#insurance'],
    ['#business', '#claims'],
    ['#motortheft'],
    ['#personalloss'],
    ['#business', '#claims'],
    ['#insurance', '#claims'],
    ['#claims', '#motortheft'],
    ['#motortheft'],
    ['#business', '#insurance'],
    ['#motortheft', '#claims'],
    ['#personalloss', '#claims'],
    ['#insurance'],
    ['#claims'],
    ['#motortheft'],
    ['#business', '#insurance'],
    ['#insurance']
]

# Convert hashtags to a binary format using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(hashtags)

# Vectorize questions using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(questions)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest classifier with One-vs-Rest strategy
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
ovr_classifier = MultiOutputClassifier(rf_classifier)

# Train the model
ovr_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = ovr_classifier.predict(X_test)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Predict hashtags for a new question
def predict_hashtags_rf(model, vectorizer, new_question, mlb):
    # Vectorize the new question
    X_new = vectorizer.transform([new_question])
    
    # Predict using the trained model
    y_pred_new = model.predict(X_new)
    
    # Convert predictions back to hashtag labels
    predicted_hashtags = mlb.inverse_transform(y_pred_new)
    
    return predicted_hashtags

# Example: Predicting hashtags for a new question
new_question = "Did you renew your insurance this year?"
predicted_hashtags = predict_hashtags_rf(ovr_classifier, tfidf_vectorizer, new_question, mlb)

print(f"Predicted Hashtags for the question '{new_question}': {predicted_hashtags}")
