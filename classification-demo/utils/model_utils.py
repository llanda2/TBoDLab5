# utils/model_utils.py

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def train_svm_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    # Initialize and train the SVM classifier
    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    # Predict probabilities and labels
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, y_score, y_pred

def compute_roc(y_test, y_score):
    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score
