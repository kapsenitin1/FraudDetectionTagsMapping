import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
import torch.optim as optim
import numpy as np

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

# MultiLabelBinarizer to convert hashtags to multi-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(hashtags)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizing the input text
inputs = tokenizer(questions, padding=True, truncation=True, return_tensors='pt')

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(inputs['input_ids'], y, test_size=0.2, random_state=42)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
def train_model(model, X_train, y_train, epochs=3):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train, labels=torch.tensor(y_train).float())
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Train the model
train_model(model, X_train, y_train)

# Testing
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    logits = outputs.logits
    predictions = (logits > 0).int()

# Evaluate the results
accuracy = accuracy_score(y_test, predictions.numpy())
f1 = f1_score(y_test, predictions.numpy(), average='micro')

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')

def predict_hashtags(model, tokenizer, new_question, mlb):
    # Tokenize the new question
    inputs = tokenizer(new_question, return_tensors='pt', padding=True, truncation=True)

    # Predict using the trained model
    with torch.no_grad():
        outputs = model(inputs['input_ids'])
        logits = outputs.logits
        # Convert logits to binary predictions (thresholding at 0 to get multi-labels)
        predictions = (logits > 0).int().cpu().numpy()

    # Convert predictions back to the original hashtag labels
    predicted_hashtags = mlb.inverse_transform(predictions)

    return predicted_hashtags

# New question to test
new_question = "Did you renew your insurance this year?"

# Predict the hashtags for the new question
predicted_hashtags = predict_hashtags(model, tokenizer, new_question, mlb)

# Print the predicted hashtags
print(f"Predicted Hashtags for the question '{new_question}': {predicted_hashtags}")