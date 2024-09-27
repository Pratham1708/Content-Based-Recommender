Movie Recommendation System
Overview
This project is a movie recommendation system that uses content-based filtering and actor similarity to suggest movies to users. The system is built using Python and Streamlit, and it uses the IMDB movie dataset to train the recommendation models.

Features
Content-based filtering: recommends movies based on their genres, directors, and casts.
Actor similarity: recommends movies based on the actors who have worked together in other movies.
User-friendly interface: allows users to select a movie and get recommendations in real-time.
Loading spinner: adds a loading animation to the interface to improve user experience.
Requirements
Python 3.12 or later
Streamlit 1.14 or later
Pandas 1.4 or later
NumPy 1.23 or later
Scikit-learn 1.1 or later
Installation
Clone the repository using git clone https://github.com/your-username/movie-recommendation-system.git
Install the required packages using pip install -r requirements.txt
Run the application using streamlit run app.py
Usage
Select a movie from the dropdown list.
Click the "Get Recommendations" button to get a list of recommended movies.
The recommended movies will be displayed in two columns: actor-based recommendations and content-based recommendations.
Dataset
The IMDB movie dataset is used to train the recommendation models. The dataset contains information about movies, including their genres, directors, casts, and ratings.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

Acknowledgments
The IMDB movie dataset is used to train the recommendation models.
The Streamlit library is used to build the user-friendly interface.
The Scikit-learn library is used to implement the content-based filtering and actor similarity algorithms.
