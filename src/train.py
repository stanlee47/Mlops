import pandas as pd
import os
import yaml
import pickle as pk
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Load config from params.yaml
path = yaml.safe_load(open('params.yaml'))['train']

def hyperparameter_tuning(x_train, y_train):
    best_model = None
    best_score = 0
    
    models = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'SVC': {
            'model': SVC(probability=True),
            'params': {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf']
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        }
    }

    for model_name, model_info in models.items():
        print(f"Training {model_name}...")
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=3,
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)
        accuracy = grid_search.score(x_train, y_train)

        print(f"{model_name} training accuracy: {accuracy}")

        if accuracy > best_score:
            best_model = grid_search.best_estimator_
            best_score = accuracy
            print(f"New best model: {model_name} with accuracy {best_score}")

    return best_model, best_score


def train_model():
    data = pd.read_csv(path['data'])
    target = 'Purchase_Made'  # Updated target column

    X = data.drop(columns=[target])
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model, best_score = hyperparameter_tuning(x_train, y_train)

    print(f"\nBest Model Final Selection:\n{best_model}\nWith Score: {best_score}")

    # Save the best model to specified path
    model_path = path['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        pk.dump(best_model, f)

    print(f"Model saved successfully at: {model_path}")


if __name__ == '__main__':
    train_model()
    print("\nTraining is Completed Successfully.")
