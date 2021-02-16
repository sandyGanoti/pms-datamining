from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    precision_score, f1_score, recall_score, precision_score, auc
)


def main():

    models = [
        RandomForestClassifier(n_estimators=200, random_state=0),
        # linear used because linear should scale better to large numbers of samples.
        LinearSVC(),
    ]

    CV = 5
    accuracies = cross_val_score(model, features, labels, scoring="accuracy", cv=CV)


if __name__ == "__main__":
    main()


# F-Measure -- f1_score(y_true,y_pred)
# Recall -- recall_score(y_true,y_pred)
# Precision -- precision_score(y_true,y_pred)
# AUC -- auc
# Accuracy ??? p = rf.predict_proba( test_x )
# auc = AUC( test_y, p[:,1] )
