from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def train_svm_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    # Initialize and train the SVM classifier
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X_train, y_train)
    return svm


def evaluate_model(model, X_test, y_test):
    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm
