import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

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
    return pd.read_csv("./data/ratings.csv")


def load_course_sims():
    return pd.read_csv("./data/sim.csv")


def load_courses():
    df = pd.read_csv("./data/course_processed.csv")
    df["TITLE"] = df["TITLE"].str.title()
    return df


def load_courses_genres():
    return pd.read_csv("./data/course_genre.csv")


def load_bow():
    return pd.read_csv("./data/courses_bows.csv")


def load_user_profiles():
    return pd.read_csv("./data/user_profile.csv")


def load_PCA_clustered_user_profiles():
    return pd.read_csv("./data/user_profile_clustered_pca.csv")


def load_clustered_user_profiles():
    return pd.read_csv("./data/user_profile_clustered.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        ratings_df = load_ratings()
        new_id = ratings_df["user"].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict["user"] = users
        res_dict["item"] = new_courses
        res_dict["rating"] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("./data/ratings.csv", index=False)

        profile_df = load_user_profiles()
        course_genres_df = load_courses_genres()
        ratings = load_ratings()
        courses = ratings[ratings["user"] == new_id]["item"]
        C = (
            course_genres_df[course_genres_df["COURSE_ID"].isin(courses)]
            .iloc[:, 2:]
            .to_numpy()
        )

        u0 = np.array([3.0] * len(courses))
        u0weights = np.matmul(u0, C)
        u0profile = pd.DataFrame(
            u0weights.reshape(1, 14), columns=profile_df.columns[1:]
        )

        u0profile.insert(0, "user", new_id)
        updated_profile = pd.concat([profile_df, u0profile]).reset_index(drop=True)
        updated_profile.to_csv("./data/user_profile.csv", index=False)

        return new_id


def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(["doc_index", "doc_id"]).max().reset_index(drop=False)
    idx_id_dict = grouped_df[["doc_id"]].to_dict()["doc_id"]
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(
    idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix
):

    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def user_profile_reommendations(idx_id_dict, enrolled_course_ids, user_vector):
    course_genres_df = load_courses_genres()
    all_courses = set(idx_id_dict.values())
    unknown_courses = all_courses.difference(set(enrolled_course_ids))
    unknown_courses_df = course_genres_df[
        course_genres_df["COURSE_ID"].isin(unknown_courses)
    ]
    unknown_courses_ids = unknown_courses_df["COURSE_ID"].values

    rows = set(course_genres_df["COURSE_ID"]).intersection(set(unknown_courses_ids))
    idx = [(x in rows) for x in course_genres_df["COURSE_ID"].values]
    recommendation_scores = np.dot(
        course_genres_df[idx].iloc[:, 2:].values, user_vector.T
    )
    res = {}
    for i in range(0, len(unknown_courses_ids)):
        res[unknown_courses_ids[i]] = recommendation_scores[i][0]
    sorted_dict = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict


def content_clustering(courses_clusters, labelled, user_id):
    recommended = {}
    union_courses = set()
    d_user = labelled[labelled["user"] == user_id]
    cluster = d_user["cluster"].values[0]
    all_courses_cluster = courses_clusters[courses_clusters["cluster"] == cluster]
    all_courses_cluster = (
        all_courses_cluster.sort_values(by="enrollments").iloc[:, 1].values
    )
    user_courses = d_user["item"].values
    new_courses = list(
        set(all_courses_cluster).difference(
            set(all_courses_cluster).intersection(set(user_courses))
        )
    )
    recommended[user_id] = new_courses
    union_courses = union_courses.union(set(new_courses))
    avail = list(union_courses)
    user_recommendations = pd.DataFrame(
        [
            [user] + [courses.count(c) for c in avail]
            for user, courses in recommended.items()
        ],
        columns=["user"] + avail,
    )
    ll = list(user_recommendations.columns[1:])
    recom = {ll[i]: len(ll) - i for i in range(len(ll))}
    return recom


# Model training
def train(model_name, params):
    if model_name == models[1]:
        pass
    elif model_name == models[2]:
        profile_df = load_user_profiles()
        ratings_df = load_ratings()
        feature_names = list(profile_df.columns[1:])
        scaler = StandardScaler()
        profile_df[feature_names] = scaler.fit_transform(profile_df[feature_names])

        features = profile_df.loc[:, profile_df.columns != "user"]
        user_ids = profile_df.loc[:, profile_df.columns == "user"]

        n_clusters = params["n_clusters"]

        kmeans = KMeans(n_init="auto", n_clusters=n_clusters)
        kmeans.fit_predict(features)

        clustered_users = user_ids.join(
            pd.DataFrame(kmeans.labels_, columns=["cluster"])
        )
        labelled = pd.merge(clustered_users, ratings_df, on="user")
        labelled.to_csv("./data/user_profile_clustered.csv", index=False)

    elif model_name == models[3]:
        profile_df = load_user_profiles()
        ratings_df = load_ratings()
        feature_names = list(profile_df.columns[1:])
        scaler = StandardScaler()
        profile_df[feature_names] = scaler.fit_transform(profile_df[feature_names])

        features = profile_df.loc[:, profile_df.columns != "user"]
        user_ids = profile_df.loc[:, profile_df.columns == "user"]

        n_clusters = params["n_clusters"]
        pca = PCA(n_components=n_clusters)
        features_red = pca.fit_transform(features)
        rename_pc = ["pca" + str(i) for i in range(1, n_clusters + 1)]
        merged_pca_df = user_ids.join(
            pd.DataFrame(features_red, columns=rename_pc)
        ).reset_index(drop=True)

        kmeans = KMeans(n_init="auto", n_clusters=n_clusters)
        kmeans.fit_predict(merged_pca_df)

        clustered_users = user_ids.join(
            pd.DataFrame(kmeans.labels_, columns=["cluster"])
        )
        labelled = pd.merge(clustered_users, ratings_df, on="user")
        labelled.to_csv("./data/user_profile_clustered_pca.csv", index=False)

    elif model_name == models[4]:
        pass
    elif model_name == models[5]:
        pass
    elif model_name == models[6]:
        pass
    elif model_name == models[7]:
        pass
    elif model_name == models[8]:
        pass
    pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}
    for user_id in user_ids:
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df["user"] == user_id]
            enrolled_course_ids = user_ratings["item"].to_list()
            res = course_similarity_recommendations(
                idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix
            )
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        elif model_name == models[1]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df["user"] == user_id]
            enrolled_course_ids = user_ratings["item"].to_list()
            user_profile = load_user_profiles()
            user_vector = (
                user_profile[user_profile["user"] == user_id].iloc[:, 1:].to_numpy()
            )
            res = user_profile_reommendations(
                idx_id_dict, enrolled_course_ids, user_vector
            )
            for key, score in res.items():
                users.append(user_id)
                courses.append(key)
                scores.append(score)

        elif model_name == models[2]:
            labelled = load_clustered_user_profiles()
            courses_clusters = labelled[["item", "cluster"]]
            courses_clusters["count"] = [1] * len(courses_clusters)
            courses_clusters = (
                courses_clusters.groupby(["cluster", "item"])
                .agg(enrollments=("count", "sum"))
                .reset_index()
            )
            res = content_clustering(courses_clusters, labelled, user_id)
            for key, score in res.items():
                users.append(user_id)
                courses.append(key)
                scores.append(score)

        elif model_name == models[3]:
            labelled = load_PCA_clustered_user_profiles()
            courses_clusters = labelled[["item", "cluster"]]
            courses_clusters["count"] = [1] * len(courses_clusters)
            courses_clusters = (
                courses_clusters.groupby(["cluster", "item"])
                .agg(enrollments=("count", "sum"))
                .reset_index()
            )

            res = content_clustering(courses_clusters, labelled, user_id)

            for key, score in res.items():
                users.append(user_id)
                courses.append(key)
                scores.append(score)
        elif model_name == models[4]:
            pass
        elif model_name == models[5]:
            pass
        elif model_name == models[6]:
            pass
        elif model_name == models[7]:
            pass
        elif model_name == models[8]:
            pass

    res_dict["USER"] = users
    res_dict["COURSE_ID"] = courses
    res_dict["SCORE"] = scores
    res_df = pd.DataFrame(res_dict, columns=["USER", "COURSE_ID", "SCORE"])
    return res_df.head(params["top_courses"])
