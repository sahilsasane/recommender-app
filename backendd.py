import pandas as pd

models = (
    "Course Similarity",
    "User Profile",
    "Clustering",
    "Clustering with PCA",
    "KNN",
    "NMF",
    "Neural Network",
    "Regression with Embedding Features",
    "Classification with Embedding Features",
)


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df["TITLE"] = df["TITLE"].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def course_similarity():
    pass


def user_profile():
    pass


def clustering():
    pass


def clusteringPCA():
    pass


def knn():
    pass


def NMF():
    pass


def neural_network():
    pass


def regression_with_encoding_features():
    pass


def classification_with_embedding_features():
    pass
