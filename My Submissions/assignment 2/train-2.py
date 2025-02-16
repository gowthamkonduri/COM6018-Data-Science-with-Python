import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load and preprocess data from a given file."""
    data = joblib.load(file_path)
    X = data['data']  # Keep data in original flattened format
    y = data['target']
    return X, y



def augment_data(X, y):
    """Apply simple data augmentation (flipping) while keeping dimensions consistent."""
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        img_pair = X[i].reshape(2, 62, 47)  # Reshape into image pairs
        flipped_pair = np.flip(img_pair, axis=2)  # Horizontal flip
        augmented_X.append(flipped_pair.flatten())  # Flatten back to original shape
        augmented_y.append(y[i])

    augmented_X = np.vstack([X, np.array(augmented_X)])  # Combine original and augmented data
    augmented_y = np.hstack([y, np.array(augmented_y)])  # Combine labels

    return augmented_X, augmented_y


def tune_hyperparameters(X_train, y_train):
    """Perform Grid Search for SVM hyperparameters."""
    param_grid = {
        'svc__C': [1, 10, 100],
        'svc__gamma': ['scale', 0.01, 0.1],
    }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=200, random_state=42)),
        ('svc', SVC(kernel='rbf', random_state=42, probability=True))
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(y_true, y_pred, y_scores):
    """Evaluate model using multiple metrics."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Compute ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])  # Use probability for the positive class
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.2f}")

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def main(train_data_path, model_save_path):
    # Load and preprocess data
    X, y = load_data(train_data_path)

    # Augment data
    X_augmented, y_augmented = augment_data(X, y)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

    # Train and tune the model
    print("Tuning hyperparameters...")
    model = tune_hyperparameters(X_train, y_train)

    # Evaluate the model
    print("Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    y_val_scores = model.predict_proba(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Additional Metrics
    evaluate_model(y_val, y_val_pred, y_val_scores)

    # Save the trained model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python train.py <TRAIN_DATA_FILE> <MODEL_FILE>")
    else:
        train_data_path = sys.argv[1]
        model_save_path = sys.argv[2]
        main(train_data_path, model_save_path)