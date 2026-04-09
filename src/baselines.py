"""
Baseline models: TF-IDF + Logistic Regression / Linear SVM.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42


def build_mnb_pipeline(max_features=50000, ngram_range=(1, 2)):
    """TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB()),
    ])


def build_logreg_pipeline(max_features=50000, ngram_range=(1, 2)):
    """TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            C=1.0,
            solver="lbfgs",
        )),
    ])


def build_svm_pipeline(max_features=50000, ngram_range=(1, 2)):
    """TF-IDF + Linear SVM pipeline (with calibration for probability estimates)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(
                max_iter=2000,
                random_state=RANDOM_SEED,
                C=1.0,
            ),
            cv=3,
        )),
    ])
