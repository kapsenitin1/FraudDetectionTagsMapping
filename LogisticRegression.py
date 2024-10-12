import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# Sample Data
questions = [
    "Did you renew your insurance this year?", 
    "Did you raise a complaint about your lost vehicle?", 
    "Have you made any business-related claims recently?", 
    "Are you self-employed or a business owner?", 
    "Have you experienced a personal loss in the last 12 months?", 
    "Did your vehicle get damaged in an accident?", 
    "Have you filed any insurance claims for property damage?", 
    "Is your business facing financial difficulties?", 
    "Did you recently submit a claim for theft of property?", 
    "Have you had any medical expenses in the last year?", 
    "Did you report your stolen vehicle?", 
    "Are you involved in any legal disputes with the insurance company?", 
    "Did you claim insurance for medical expenses?", 
    "Have you reported any losses related to natural disasters?", 
    "Have you been denied an insurance claim before?", 
    "Did your business suffer any equipment theft recently?", 
    "Are you currently covered under health insurance?", 
    "Have you been involved in a car accident recently?", 
    "Did your business file a claim for fire damage?", 
    "Have you experienced a personal injury at work?"
]

hashtags = [
    ['#insurance', '#claims'], 
    ['#motertheft', '#vehicle', '#stolen'], 
    ['#business', '#claims'], 
    ['#business'], 
    ['#personalloss'], 
    ['#vehicle', '#accident', '#claims'], 
    ['#propertydamage', '#claims', '#insurance'], 
    ['#business', '#financialdifficulties'], 
    ['#theft', '#claims', '#propertydamage'], 
    ['#medical', '#health'], 
    ['#stolen', '#motertheft'], 
    ['#legaldispute', '#insurance'], 
    ['#insurance', '#medical'], 
    ['#naturaldisaster', '#claims'], 
    ['#deniedclaim', '#claims'], 
    ['#business', '#equipmenttheft', '#theft'], 
    ['#health', '#insurance'], 
    ['#caraccident', '#accident', '#vehicle'], 
    ['#business', '#fire', '#claims'], 
    ['#injury', '#personalloss']
]

# Convert hashtags to multi-label format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(hashtags)

# TF-IDF Vectorization for questions
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(questions)

# Remove labels with insufficient samples
y_sum = y.sum(axis=0)
min_samples = 3  # Adjust based on dataset size
non_single_class_labels = np.where((y_sum >= min_samples) & (y_sum <= y.shape[0] - min_samples))[0]
y = y[:, non_single_class_labels]
mlb.classes_ = mlb.classes_[non_single_class_labels]

# Logistic Regression with One-vs-Rest Strategy
lr = LogisticRegression(max_iter=2000)

# Hyperparameter tuning
param_grid = {'estimator__C': [0.01, 0.1, 1, 10, 100]}
ovr_classifier = MultiOutputClassifier(lr)

# Use MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=3, random_state=42, shuffle=True)

grid_search = GridSearchCV(ovr_classifier, param_grid, cv=mskf, scoring='f1_micro', n_jobs=-1, error_score='raise')
grid_search.fit(X, y)

# Best parameters and model
best_classifier = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Function to predict hashtags for a new question
def predict_hashtags_ovr(model, vectorizer, new_question, mlb):
    # Vectorize the new question
    X_new = vectorizer.transform([new_question])
    
    # Predict using the trained model
    y_pred_new = model.predict(X_new)
    
    # Convert predictions back to hashtag labels
    predicted_hashtags = mlb.inverse_transform(y_pred_new)
    
    return predicted_hashtags

# Predicting hashtags for a new question
new_question = "Did you renew your insurance this year?"
predicted_hashtags = predict_hashtags_ovr(best_classifier, tfidf_vectorizer, new_question, mlb)

print(f"Predicted Hashtags for the question '{new_question}': {predicted_hashtags}")
