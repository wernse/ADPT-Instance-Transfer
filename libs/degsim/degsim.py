import pandas as pd

from sklearn.externals import joblib

#Pearsons R
from scipy.spatial.distance import correlation
from sklearn.neighbors import NearestNeighbors
from scipy.stats.stats import pearsonr

model_name = './models/nn.joblib'
data_model_name = './models/nn.xlsx'
neighbours = 15


def train_degsim_model(X_train):
    neigh = NearestNeighbors(n_neighbors=neighbours + 1, metric='correlation')
    neigh.fit(X_train)
    save_model(neigh, X_train)


#need to load in the data for the test set!
#for the correlation to be calculated against the target
def test_degsim_model(normalised_df):
    clf, train_data = load_model()
    degsim_df = pd.DataFrame()
    for row in normalised_df.iterrows():
        target_index = row[0]
        # print("target_index", target_index)
        target = normalised_df.loc[[target_index]]
        rng = clf.kneighbors(target)
        #remove the neighbour with irself
        neighbours_index = list(rng[1][0])
        if target_index in neighbours_index:
            neighbours_index.remove(target_index)

        target = normalised_df.loc[target_index]
        degsim = calc_degsim_single(target, train_data, neighbours_index)
        degsim_df.loc[target_index, 'degsim'] = degsim
    return degsim_df


def calc_degsim_single(target, train_data, neighbours_index):
    target_degsim = 0
    for i in enumerate(neighbours_index):
        index = i[1]
        # print("index", index)
        neighbour = train_data.iloc[index]
        correlation = pearsonr(target, neighbour)
        # print("correlation", correlation)
        target_degsim = target_degsim + correlation[0]
    target_degsim = target_degsim / len(neighbours_index)
    return target_degsim


def save_model(model, data):
    #save model
    joblib.dump(model, model_name)

    #save data
    writer = pd.ExcelWriter(data_model_name)
    data.to_excel(writer)
    writer.save()


def load_model():
    clf = joblib.load(model_name)
    train_data = pd.read_excel(data_model_name)

    return clf, train_data
