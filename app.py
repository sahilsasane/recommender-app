import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

st.set_page_config(
    page_title="Course Recommender", layout="wide", initial_sidebar_state="expanded"
)


@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()


def init_app():
    with st.spinner("Loading Datasets"):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    st.success("Datasets loaded successfully")
    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")
    return 1


def train(model_name, params):
    pass


st.sidebar.title("Personalized recommender")
selected_courses = init_app()

st.sidebar.header("1. Select recommendation models")
model_selection = st.sidebar.selectbox("select model:", backend.models)

params = {}
st.sidebar.subheader("2. Tune Hyper-parameters: ")
if model_selection == backend.models[0]:
    top_courses = st.sidebar.slider(
        "Top courses", min_value=0, max_value=100, value=10, step=1
    )
    course_sim_threshold = st.sidebar.slider(
        "Course Similarity Threshold %", min_value=0, max_value=100, value=50, step=10
    )
    params["top_courses"] = top_courses
    params["sim_threshold"] = course_sim_threshold

st.sidebar.subheader("3. Training: ")
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text("")

if training_button:
    train(model_selection, params)
