import pandas as pd

model_name = './models/normalised_range.xlsx'


def train_normalise_model(X_train):
    #need to normalise
    #trained model - need this this for future normalisation
    col_max = X_train.max()
    col_min = X_train.min()
    normalised_range_df = pd.DataFrame([col_max, col_min])
    writer = pd.ExcelWriter(model_name)
    normalised_range_df.to_excel(writer)
    writer.save()


def test_normalise_model(clean_df):
    df = pd.read_excel(model_name)
    col_max = df.loc[0]
    col_min = df.loc[1]

    normalised_df = clean_df.copy()
    normalised_df = normailise_dataframe(normalised_df, col_max, col_min)
    return normalised_df


def format_dataframe(data):
    new_df = data.copy()
    print(data.columns)
    data_df = data.drop(
        ['tweet_id', 'tweet_text', 'created_at', 'county_code', 'url'], axis=1)
    data_df.user_description_exists = data_df.user_description_exists.astype(
        int)
    data_df.user_is_verified = data_df.user_is_verified.astype(int)
    # data_df = data_df.drop([
    #     'user_description_exists', 'user_following_count',
    #     'domain_url_dot_count'
    # ],
    #                        axis=1)
    return data_df


    #normalise
def normailise_dataframe(data_df, max_df, min_df):
    return (data_df - min_df) / (max_df - min_df)


def drop_labels(data_df):
    return data_df.drop(['tweet_deleted'], axis=1)
