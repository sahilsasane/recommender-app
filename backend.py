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


joined_df_with_clusters = 0


# Model training
def train(model_name, params):
    if model_name == models[1]:
        pass
    elif model_name == models[2]:
        cluster = params["clusters"]

        sims_df = load_course_sims()
        course_processed = load_courses()
        course_ids = pd.DataFrame(course_processed.loc[:, "COURSE_ID"])
        pca = PCA(n_components=cluster)
        pca_result = pca.fit_transform(sims_df)
        merged = course_ids.join(pd.DataFrame(pca_result)).reset_index()
        pc_rename = {i: f"PC{i}" for i in range(len(merged.columns) - 1)}
        merged.rename(
            columns=pc_rename,
            inplace=True,
        )
        kmeans = KMeans(n_clusters=cluster)
        print(merged.iloc[:, 2:])
        clusters = kmeans.fit_predict(merged.iloc[:, 2:])
        joined_df_with_clusters = (
            course_processed.join(pd.DataFrame(clusters))
            .reset_index()
            .drop(["index", "TITLE", "DESCRIPTION"], axis=1)
        )
        print(joined_df_with_clusters.head())

    elif model_name == models[3]:
        pass
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
            pass
        elif model_name == models[3]:
            pass

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
