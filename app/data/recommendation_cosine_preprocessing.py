import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import scipy.sparse

credit = pd.read_csv('recommendation_cosine/tmdb_5000_credits.csv')
movies = pd.read_csv('recommendation_cosine/tmdb_5000_movies.csv')


def preprocessing():
    """
    preprocess data used for recommendation system. See detailed version on
    https://github.com/tt0507/Mini-Project/blob/master/recommendation_system/data_preprocessing.ipynb
    :return:
    """
    global credit, movies

    # merge two data frame
    credit.columns = ['id', 'title', 'cast', 'crew']
    movies = movies.merge(credit, on='id')

    # drop homepage column and rows with overview as nan
    movies = movies.drop(columns=['homepage'])
    movies = movies.drop(movies[movies['overview'].isnull()].index.tolist())

    # manually input the release_date of the movie
    movies['release_date'].loc[movies['id'] == 380097] = "2015-03-01"

    # reset index
    movies = movies.reset_index(drop=True)

    # fill NA values of tagline column
    movies['tagline'] = movies['tagline'].fillna('')

    # get weighted_rating for each movie
    movies['score'] = movies.apply(weighted_rating, axis=1)

    # apply literal_eval to all rows in the column
    features = ['genres', 'keywords', 'production_companies', 'production_countries',
                'spoken_languages', 'cast', 'crew']
    for feature in features:
        movies[feature] = movies[feature].apply(literal_eval)

    # get name key for all columns except for crew
    for feature in features:
        if feature != 'crew':
            movies[feature] = movies[feature].apply(lambda x: [i['name'] for i in x])

    # get the directors for the crew column
    movies['crew'] = movies['crew'].apply(get_director)

    # apply tokenization method to the columns below
    movie_info = ['cast', 'keywords', 'crew', 'genres']
    for info in movie_info:
        movies[info] = movies[info].apply(tokenization)

    # apply tokenization method to the columns below
    movie_country = ["production_companies", "production_countries", "spoken_languages"]
    for column in movie_country:
        movies[column] = movies[column].apply(tokenization)

    # concatenate the tokenized columns
    movies['info'] = movies.apply(concatenate_info, axis=1)
    movies['country_info'] = movies.apply(concatenate_countries, axis=1)

    # drop unnecessary columns
    movies = movies.drop(columns=['cast', 'keywords', 'crew', 'genres',
                                  'production_companies', 'production_countries', 'spoken_languages'])

    movies = movies.drop(columns=['title_x', 'title_y', 'status', 'release_date',
                                  'original_language', 'budget', 'revenue', 'runtime'])

    # reorder index of data frame
    movies = movies.set_index('id').sort_values(by='id')

    # separate into word and numerical column for ColumnTransformer
    word_column = ['original_title', 'overview', 'tagline', 'info', 'country_info']
    num_column = ['popularity', 'vote_average', 'vote_count', 'score']
    assert len(movies.columns) == len(word_column) + len(num_column)

    # create pipeline for ColumnTransformer
    full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_column),
        ("original_title", TfidfVectorizer(), "original_title"),
        ("overview", TfidfVectorizer(), "overview"),
        ("tagline", TfidfVectorizer(), "tagline"),
        ("info", TfidfVectorizer(), 'info'),
        ("country_info", TfidfVectorizer(), "country_info")
    ])

    # fit and transform the data frame
    movie_data = full_pipeline.fit_transform(movies)

    # export data
    scipy.sparse.save_npz('recommendation_cosine/movie.npz', movie_data)  # create sparse matrix
    movies.to_csv('recommendation_cosine/movie.csv')  # create data frame


def weighted_rating(x, m=3000, C=movies['vote_average'].mean()):
    """
    create weight average based on IMDB's weighted rating formula

    R is the average rating of the movie
    v is the number of votes for the movie
    m is the minimum votes required to be listed in top 250 (which is 3000)
    C us the mean vote across the whole report
    :param x: row
    :param m:
    :param C:
    :return: weighted average score
    """
    v = x['vote_count']
    R = x['vote_average']
    return (R * v + C * m) / (v + m)


def get_director(val):
    """
    Get the director of the movie described in the column
    :param val: row of dataframe
    :return:
    """
    for i in val:
        if i['job'] == 'Director':
            return i['name']
    return "None"


def tokenization(x):
    """
    change string into lower case and combine everything into one word
    :param x: row of dataframe
    :return:
    """
    if isinstance(x, list):
        return [str.lower(word.replace(" ", "")) for word in x]
    else:
        if isinstance(x, str) and x != "None":
            return str.lower(x.replace(" ", ""))
        else:
            return str.lower(x.replace("None", ""))


def concatenate_info(x):
    """
    concatenate string in different columns into one large string separated by space
    :param x: row of dataframe
    :return:
    """
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' \
           + x['crew'] + ' ' + ' '.join(x['genres'])


def concatenate_countries(x):
    """
    concatenate string in different columns into one large string separated by space
    :param x: row of dataframe
    :return:
    """
    return ' '.join(x['production_companies']) + ' ' + ' '.join(x['production_countries']) + ' ' \
           + ' '.join(x['spoken_languages'])


if __name__ == "__main__":
    preprocessing()
