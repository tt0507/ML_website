import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def get_recommendation(title):
    """
    get recommended movie from the title
    :param title: title of the movie
    :return:
    """
    # import data
    movie_data = scipy.sparse.load_npz('../data/recommendation_cosine/movie.npz')
    movie_index = pd.read_csv('../data/recommendation_cosine/movie.csv')

    # get cosine similarity
    cosine_sim = cosine_similarity(movie_data, movie_data)

    # constant for amount of recommendation to show
    SHOW = 10

    # return the recommendation
    index = movie_index[movie_index['original_title'] == title].index
    cosine_sim_score = list(enumerate(cosine_sim[index[0]]))
    sorted_score = sorted(cosine_sim_score, key=lambda entry: entry[1], reverse=True)
    show_top = sorted_score[1:SHOW + 1]
    index_list = [index[0] for index in show_top]
    return movie_index['original_title'].iloc[index_list]


if __name__ == "__main__":
    # for testing purpose
    print(get_recommendation('The Dark Knight Rises'))
