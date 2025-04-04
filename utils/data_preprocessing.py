import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path):
    # Load Titanic dataset
    df = pd.read_csv("data/titanic.csv")

    # Select features and target variable
    features = df[['Age', 'Fare']]
    target = df['Survived']

    # Handle missing values by filling with median
    features['Age'].fillna(features['Age'].median(), inplace=True)

    # Scale features for better model performance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, target, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler
