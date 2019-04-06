import pandas as pd
import datetime
import time
from libs.db import DbClass
from libs.cluster.knn import train_knn_model, load_knn_model
from libs.nb.nb import train_nb_model, load_nb_model, load_nb_model_2
from libs.svm.svm import train_svm_model, test_svm_model
from libs.decision_tree.decision_tree import train_tree_model, load_tree_model
from libs.outlier.isoliation import train_outlier_model, load_outlier_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import Counter
from sklearn.neighbors import NearestNeighbors


class KNNData():
    def __init__(self, data):
        self.data = data
        self.scaler = self.create_scaler()

    def create_scaler(self):
        data = TweetData(self.data)
        scaler = MinMaxScaler()
        scaler.fit(data.df_formatted[data.df_formatted.columns])
        return scaler

    def create_source_domain(self, source_country):
        self.source_domain = TweetData(
            self.data[self.data["county_code"] == source_country])
        self.source_domain.normalise_data(self.scaler)

    def create_target_domain(self, target_country):
        self.target_domain = TweetData(
            self.data[self.data["county_code"] == target_country])
        self.target_domain.normalise_data(self.scaler)

    #for each point get neighbours
    def get_neighbours(self, neighbours=10):
        nbrs = NearestNeighbors(
            n_neighbors=neighbours,
            metric='euclidean').fit(self.source_domain.df_formatted)
        distances, indices = nbrs.kneighbors(self.target_domain.df_formatted)
        flat_distances = [item for sublist in distances for item in sublist]
        flat_indices = [item for sublist in indices for item in sublist]
        valid_indicies = list()
        for idx, distance in enumerate(flat_distances):
            print(distance)
            if distance > 0.01 and flat_indices[idx] not in valid_indicies:
                valid_indicies.append(flat_indices[idx])
        source_domain_knn = self.source_domain.data.iloc[valid_indicies]
        return source_domain_knn


class TweetData():
    def __init__(self, data):
        self.data = data

        self.labels = data["tweet_deleted"]
        self.format_data()

    def format_data(self):
        df_no_labels = self.data.drop(['tweet_deleted'], axis=1).copy()
        data_df = df_no_labels.drop([
            'tweet_id', 'tweet_text', 'created_at', 'county_code', 'url',
            'date'
        ],
                                    axis=1)
        data_df.user_description_exists = data_df.user_description_exists.astype(
            int)
        data_df.user_is_verified = data_df.user_is_verified.astype(int)
        self.df_formatted = data_df

    def normalise_data(self, scaler):
        self.df_formatted[self.df_formatted.columns] = scaler.transform(
            self.df_formatted[self.df_formatted.columns])

    def get_stats(self):
        return Counter(self.data["county_code"]).most_common()

    def get_phishing(self):
        return Counter(self.data["tweet_deleted"]).most_common()

    def get_predicted(self, metric):
        return Counter(self.data[metric]).most_common()

    def print(self, writer, name):
        self.print = self.data[self.data["tweet_deleted"] == True]
        self.print.to_excel(writer, name)


def main(train_countries, test_countries, instance_based=False):
    print("Running ADPT Process for Train Countries:", train_countries)
    print("Running ADPT Process for Test Countries:", test_countries)
    print("Running ADPT Process Instance Transfer:", instance_based)

    start_time = time.time()
    # Setup start date and interval
    start_date = '2018-09-01'
    dates = ['2018-11-01', '2018-12-05', '2019-01-29']
    test_dates = ['2018-11-01', '2018-12-05', '2019-01-29']

    # The string format
    format_str = '%Y-%m-%d'

    # Label
    label = "tweet_deleted"
    relabel = False

    writer = pd.ExcelWriter(
        './results/crossfoldrl{}_{}_{}_{}_{}_ib{}.xlsx'.format(
            relabel, label, train_countries[0], test_countries[0],
            len(test_countries), instance_based),
        engine='xlsxwriter',
        options={'strings_to_urls': False})

    df_all = get_data(start_date, dates[len(dates) - 1], False, label, relabel)
    df_all['date'] = df_all['created_at'].apply(
        lambda x: datetime.datetime.strptime(x[0:10], format_str))

    # KNN Data, pre populate the data, add the data as NZ data
    if instance_based:
        print("Running Instanced based transfer")
        knn_data = KNNData(df_all)
        knn_data.create_source_domain("US")
        knn_data.create_target_domain("NZ")
        source_domain_knn = knn_data.get_neighbours()

    df_stats = df_all[["date", "tweet_deleted", "county_code"]].groupby(
        ["county_code", "date"]).count()
    train_start = datetime.datetime.strptime(start_date, format_str)
    results_list = list()
    for date in dates:
        train_end = datetime.datetime.strptime(date, format_str)
        mask = (df_all['date'] >= train_start) & (
            df_all['date'] <
            train_end) & df_all["county_code"].isin(train_countries)
        df_train = df_all.loc[mask]

        if instance_based:
            df_train = pd.concat([source_domain_knn, df_train])

        train_data = TweetData(df_train)

        scaler = MinMaxScaler()
        scaler.fit(train_data.df_formatted[train_data.df_formatted.columns])
        train_data.normalise_data(scaler)

        print("Training:Outlier:Starting")
        train_outlier_model(train_data.df_formatted, train_data.labels)
        print("Training:Outlier:Stopping")
        print("Training:NB:Starting")
        train_nb_model(train_data.data["tweet_text"], train_data.labels)
        print("Training:NB:Stopping")

        #Perform the test
        test_start = start_date
        outlier, tfidf, nb = (load_outlier_model(), load_nb_model_2(),
                              load_nb_model())
        test_results = {}
        test_index = 1
        for date in test_dates:
            test_end = datetime.datetime.strptime(date, format_str)
            mask = (df_all['date'] >= test_start) & (
                df_all['date'] <
                test_end) & df_all["county_code"].isin(test_countries)
            df_test = df_all.loc[mask]

            test_data = TweetData(df_test)
            test_data.normalise_data(scaler)
            print("Testing:Ensemble:Starting")
            nb_labels, outlier_labels = ensemble(outlier, tfidf, nb,
                                                 test_data.df_formatted,
                                                 test_data.data["tweet_text"])
            adpt_labels = []
            for (idx, arg) in enumerate(nb_labels):
                adpt_labels.append(arg or outlier_labels[idx])
            test_data.data["nb"] = nb_labels
            test_data.data["svm"] = outlier_labels
            test_data.data["adpt"] = adpt_labels

            results_list.append(
                calc_metrics("nb", nb_labels, test_data.labels, test_end,
                             train_end, train_data.get_stats(),
                             test_data.get_stats(), test_data.get_phishing(),
                             train_data.get_phishing(),
                             test_data.get_predicted("nb"), test_data.data))
            results_list.append(
                calc_metrics("svm", outlier_labels, test_data.labels, test_end,
                             train_end, train_data.get_stats(),
                             test_data.get_stats(), test_data.get_phishing(),
                             train_data.get_phishing(),
                             test_data.get_predicted("svm"), test_data.data))
            results_list.append(
                calc_metrics("adpt", adpt_labels, test_data.labels, test_end,
                             train_end, train_data.get_stats(),
                             test_data.get_stats(), test_data.get_phishing(),
                             train_data.get_phishing(),
                             test_data.get_predicted("adpt"), test_data.data))
            test_start = test_end
        results_list.append(test_results)
        train_start = train_end
    df_metrics = pd.DataFrame(results_list)
    df_metrics = df_metrics.sort_values(by=['1train_date', '2test_date'])

    df_metrics.to_excel(writer, "metric")
    df_stats.to_excel(writer, "metric_stats")
    writer.save()

# y is the total cost
# x is how expensive 2x more expensive to miss a phishing tweet, ratio fp:fn
# fp x 1 : fn x 2
# fp x 1 : fn x 3

def calc_metrics(name, metric, labels, test_date, train_date, train_stats,
                 test_stats, test_phishing_count, train_phishing_count,
                 predict_count, test_data):
    df_p = test_data[test_data[name] == True]
    df_np = test_data[test_data[name] == False]
    true_positives = df_p[df_p["tweet_deleted"] == True].shape[0]
    false_positives = df_p[df_p["tweet_deleted"] == False].shape[0]
    false_negatives = df_np[df_np["tweet_deleted"] == True].shape[0]
    true_negatives = df_np[df_np["tweet_deleted"] == False].shape[0]
    if true_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    if true_positives != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0

    if precision != 0 and recall != 0:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1_score = 0

    return {
        "name": name,
        "1train_date": train_date.strftime("%Y-%m-%d"),
        "2test_date": test_date.strftime("%Y-%m-%d"),
        "f1_score": f1_score,
        "precision_score": precision,
        "recall_score": recall,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "test_phishing_count": test_phishing_count,
        "train_phishing_count": train_phishing_count,
        "predict_count": predict_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def ensemble(outlier, tfidf, nb, df, text):
    outlier_label = outlier.predict(df)
    outlier_label = list(map(lambda x: x == -1, outlier_label))
    tfidf_label = tfidf.transform(text)
    nb_label = nb.predict(tfidf_label)
    return (nb_label, outlier_label)


def get_data(start, end, sample, label, relabel):
    db = DbClass()
    conn = db.connect()
    df = db.get_features_tweet(conn, start, end, sample, label, relabel)
    conn.close()
    return df
