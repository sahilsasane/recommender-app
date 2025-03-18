# Course Recommender System

## Overview
A sophisticated course recommendation system built with Python and Streamlit that helps users discover personalized learning paths. The application uses multiple recommendation algorithms to suggest relevant courses based on user preferences and learning history.

## Features
- **Multiple Recommendation Models:**
  - Content-Based Filtering
  - User Profile-Based Recommendations
  - Clustering-Based Recommendations (with and without PCA)
  - Course Similarity Analysis

- **Interactive User Interface:**
  - Easy course selection through an interactive grid
  - Adjustable recommendation parameters
  - Real-time recommendation generation
  - Wide screen layout for better visibility

- **Advanced Analytics:**
  - Course similarity analysis
  - User profile clustering
  - Principal Component Analysis (PCA) for dimension reduction
  - Genre-based content analysis

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Key Libraries:**
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - scikit-learn: Machine learning algorithms
  - st_aggrid: Interactive grid component

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/sahilsasane/recommender-app.git
cd recommender-app
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run recommender_app.py
```

## Usage

1. **Select Courses:**
   - Browse through the available courses in the interactive grid
   - Select courses you have completed or are interested in
   - View your selected courses in the summary table

2. **Configure Recommendations:**
   - Choose a recommendation model from the sidebar
   - Adjust hyperparameters:
     - Number of top courses (1-20)
     - Similarity threshold (0-100%)
     - Number of clusters (for clustering-based models)

3. **Generate Recommendations:**
   - Click "Train Model" to process your selections
   - Click "Recommend New Courses" to get personalized suggestions

## Models Available

1. **Course Similarity-Based:**
   - Uses content-based similarity between courses
   - Configurable similarity threshold
   - Recommends courses similar to your completed ones

2. **User Profile-Based:**
   - Creates a profile based on your course preferences
   - Considers course genres and topics
   - Matches your profile with available courses

3. **Clustering-Based:**
   - Groups users with similar learning patterns
   - Recommends courses popular within your cluster
   - Optional PCA for dimension reduction

4. **Combined Approaches:**
   - Multiple models can be used for comparison
   - Different parameters can be tested for optimal results

## Directory Structure
```
recommender-app/
├── data/
│   ├── course_processed.csv
│   ├── course_genre.csv
│   ├── courses_bows.csv
│   ├── ratings.csv
│   └── user_profile.csv
├── app.py
├── backend.py
├── recommender_app.py
├── course_similarity.ipynb
└── content_user_profile.ipynb
```


## Live Demo
🔗 [Access the live application](https://course0recommender.streamlit.app/)

## License
This project is available under the MIT License.
