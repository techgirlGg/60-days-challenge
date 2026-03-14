from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# true labels
y_true = [1,0,1,1,0]

# model predictions
y_pred = [1,0,0,1,0]

# Precision
precision = precision_score(y_true, y_pred)

# Recall
recall = recall_score(y_true, y_pred)

# F1 Score
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)