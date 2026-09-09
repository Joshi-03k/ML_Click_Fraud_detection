from preprocessing import load_and_prepare_data

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


print("\n" + "=" * 60)
print("K NEAREST NEIGHBORS")
print("=" * 60)


X, y, preprocessor = load_and_prepare_data()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


pipeline = Pipeline(
    steps=[

        ("preprocessing", preprocessor),

        (
            "pca",
            PCA(
                n_components=0.95
            )
        ),

        (
            "classifier",
            KNeighborsClassifier()
        )
    ]
)


param_grid = {

    "pca__n_components": [
        0.90,
        0.95
    ],

    "classifier__n_neighbors": [
        3,
        5,
        7
    ],

    "classifier__weights": [
        "uniform",
        "distance"
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\nStarting GridSearchCV...")
print("5-Fold Cross Validation")
print("Total combinations: 12")
print("Please wait...\n")


grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


grid.fit(
    X_train,
    y_train
)


print("\nGridSearch completed!")


print("\nBest Parameters:")
print(grid.best_params_)


print("\nBest CV F1 Score:")
print(
    round(
        grid.best_score_,
        4
    )
)


y_pred = grid.best_estimator_.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.title(
    "KNN Confusion Matrix"
)

plt.tight_layout()
plt.show()


print("\nKNN completed!")