import pandas as pd
import numpy as np
from libs.helpers.helpers import format_dataframe, drop_labels


# Takes in the actual dfs and based on the first df it cuts the 2nd one
def calc_t_test(df, df2):
    from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

    a = [
        0.750761731, 0.904479404, 0.901527576, 0.812183, 0.755201226,
        0.902187431, 0.904584794, 0.739829183
    ]
    b = [
        0.897143924, 0.898811096, 0.909470034, 0.899850243, 0.897295619,
        0.900187773, 0.900507107, 0.880380128
    ]
    wil = wilcoxon(a, b)
    print("wil", wil)
    #split the df into labelled data
    df = format_dataframe(df.copy())
    df = drop_labels(df)

    df2 = format_dataframe(df2.copy())
    df2 = drop_labels(df2)
    print("df.mean()")
    print(df.mean())
    print("df2.mean()")
    print(df2.mean())
    for inx, feature in enumerate(df.columns):
        print(feature)
        t_test = ttest_ind(df[feature], df2[feature])
        try:
            man = mannwhitneyu(df[feature], df2[feature])
        except:
            pass

        print(man)

    # print("results", results)
    # print(total_divergence)
    # return total_divergence


# Takes in the actual dfs and based on the first df it cuts the 2nd one
def calc_pair_divergence(df, df2):
    print("calc_pair_divergence")
    #split the df into labelled data
    # df = format_dataframe(df.copy())
    # df = drop_labels(df)

    # df2 = format_dataframe(df2.copy())
    # df2 = drop_labels(df2)
    df = df.copy()
    df2 = df2.copy()
    results = {}
    total_divergence = 0
    #domain_redirect_count
    for inx, feature in enumerate(df.columns):
        kl_diverg = calc_kl_divergence(df, df2, feature)
        total_divergence = total_divergence + kl_diverg
        results[feature] = kl_diverg
    # return pd.DataFrame([results])
    # print("results", results)
    # print(total_divergence)
    return total_divergence


def calc_kl_divergence(df, df2, target_var):
    threshold_var = target_var + '_bin'
    bins = calc_bins(df[target_var])
    df[threshold_var] = pd.cut(df[target_var], bins)
    thresholds = df[threshold_var].cat.categories
    df2[threshold_var] = pd.cut(df2[target_var], thresholds)
    # print(thresholds)
    # print(len(bins) - 1)
    max_threshold = thresholds[len(bins) - 2]
    # print("max_threshold.right", max_threshold.right)
    #get the null then apply if greater or less than the range, assign largest and smallest
    df2.loc[(df2[threshold_var].isnull()) &
            (df2[target_var].astype(int) >= max_threshold.right
             ), threshold_var] = max_threshold
    # print(
    #     "df2[target_var] > max_threshold.right",
    #     df2.loc[(df2[target_var] > max_threshold.right
    #              ), [target_var, threshold_var]])
    # print("max_threshold.right", max_threshold.right)
    # print(
    #     "NULL GREATER THAN",
    #     df2.loc[(df2[threshold_var].isnull()) &
    #             (df2[target_var] > max_threshold.right), threshold_var])
    min_threshold = thresholds[0]
    df2.loc[(df2[threshold_var].isnull()) &
            (df2[target_var].astype(int) <= min_threshold.left
             ), threshold_var] = min_threshold
    # print("min_threshold.left", min_threshold.left)
    # print(
    #     "NULL LESS THAN",
    #     df2.loc[(df2[threshold_var].isnull()) &
    #             (df2[target_var] < min_threshold.left), threshold_var])

    a = df.groupby(threshold_var).size() / df.shape[0]
    b = df2.groupby(threshold_var).size() / df2.shape[0]
    return KL(a, b)


def myRange(start, end, step):
    i = start
    while i < end:
        yield i
        i += step
    yield end


def calc_bins(data):
    bins = []
    data_mean = data.mean()
    data_std = data.std()
    if data_std == 0:
        return [0, 1]
    for i in myRange(-3, 3, 0.5):
        band = data_mean + (i * data_std)
        bins.append(band)
    return bins


# def KL(p, q):
#     p = np.asarray(p, dtype=np.float)
#     q = np.asarray(q, dtype=np.float)

#     return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def KL(P, Q):
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence