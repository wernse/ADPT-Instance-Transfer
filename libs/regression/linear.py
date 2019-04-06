# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from libs.helpers.helpers import test_normalise_model, drop_labels
"""
    Given a dataframe and variable pair
    Get the beta and intercept
"""

model_name = './models/linear_rules.xlsx'
rules_count = 6


# generates the linear relationships
def train_linear_rules(X_train):
    print("train_linear_rules")
    #split the df into labelled data
    deleted = X_train.mask('tweet_deleted', True)
    deleted = drop_labels(deleted)
    deleted = test_normalise_model(deleted)
    deleted = drop_boolean_vars(deleted)

    print("deleted_df len:", deleted.shape[0])

    not_deleted = X_train.mask('tweet_deleted', False)
    not_deleted = drop_labels(not_deleted)
    not_deleted = test_normalise_model(not_deleted)
    not_deleted = drop_boolean_vars(not_deleted)
    print("not_deleted_df len:", not_deleted.shape[0])

    deleted_df = generate_linear_frame(deleted)
    not_deleted_df = generate_linear_frame(not_deleted)

    rule_list = list()
    for index, i in enumerate(range(deleted_df.shape[0])):
        # get the deleted and not_deleted betas
        deleted_row = deleted_df.iloc[index]
        not_deleted_row = not_deleted_df.iloc[index]
        # get the positive/negative rules
        # is the beta for phishing positive?
        if (deleted_row['B'] > 0 and not_deleted_row['B'] < 0) or (
                deleted_row['B'] < 0 and not_deleted_row['B'] > 0):
            # print(deleted_row['X'], deleted_row['Y'])
            # print(deleted_row['B'], not_deleted_row['B'])

            # absolute_beta_diff = min(
            #     abs(deleted_row['B']), abs(not_deleted_row['B']))
            absolute_beta_diff = abs(deleted_row['B'] - not_deleted_row['B'])

            rule_list.append([
                deleted_row['X'], deleted_row['Y'], deleted_row['B'],
                not_deleted_row['B'], absolute_beta_diff
            ])

    # sort
    rule_df = pd.DataFrame(
        rule_list,
        columns=['X', 'Y', 'del_beta', 'not_del_beta', 'abs_max_beta'])
    rule_df = rule_df.sort_values(by='abs_max_beta', ascending=False)
    rules = rule_df.iloc[0:rules_count]

    writer = pd.ExcelWriter(model_name)
    rules.to_excel(writer)
    writer.save()


# label_boolean returns if the beta is above 0 if label_boolean
def generate_linear_frame(data_df, label_boolean=False):
    #iterate through all combinations and get the pairs_betas for deleted and not deleted
    beta_list = list()
    for outer_inx, x_var in enumerate(data_df.columns):
        for inx, y_var in enumerate(data_df.columns):
            if x_var == y_var:
                pair_result = [x_var, y_var, 0]
                beta_list.append(pair_result)
                continue

            pair_result = get_linear(data_df, x_var, y_var, label_boolean,
                                     False)
            beta_list.append(pair_result)
    df = pd.DataFrame(beta_list, columns=['X', 'Y', 'B'])
    # df = df.pivot(index='X', columns='Y', values='B')
    return df


# linear regression is on the whole cluster
def test_linear_model(X_test):
    rule_result = list()
    df = pd.read_excel(model_name)
    # print(df)
    is_phishing_count = 0
    # print("count", X_test.shape[0])
    for index, i in enumerate(range(df.shape[0])):
        # get the deleted and not_deleted betas
        pair = df.iloc[index]
        # print("pair", pair)

        x_var = pair['X']
        y_var = pair['Y']

        # If True then negative beta is phishing
        # del_beta	not_del_beta
        # -2.2716	0.65645
        # threshold
        threshold = (pair['del_beta'] + pair['not_del_beta']) / 2
        is_neg_beta_del = pair['del_beta'] < 0
        # print("threshold", threshold)
        # print("x_var", x_var)
        # print("y_var", y_var)
        # print("------------------")

        # print("is_neg_beta_del", is_neg_beta_del)
        #calculate betas
        intercept, beta = get_linear(X_test, x_var, y_var, False, True)

        subset_x = X_test[[x_var, y_var]]
        # print(subset_x.head())
        # print("beta", beta)
        # print("std:", subset_x.std())
        print("------------------")
        is_phising = beta < threshold and is_neg_beta_del
        if is_phising:
            is_phishing_count = is_phishing_count + 1
        else:
            is_phishing_count = is_phishing_count - 1
        rule_result.append(is_phising)

    label = 0
    #assess based on count assign the label
    if is_phishing_count > 0:
        label = 1
    elif is_phishing_count < 0:
        label = -1

    # "Phishing": 1
    # "Legitimate": -1
    # "Indeterministic": 0
    #return the results of each rule and the final label
    return rule_result, label


def get_linear(features_df,
               subset_x_var,
               subset_y_var,
               label_boolean=False,
               linear_stats=False):
    #create subset
    subset_x = features_df[[subset_x_var]]
    subset_y = features_df[[subset_y_var]]
    #run linear regression
    intercept, beta = run_linear(subset_x, subset_y)
    if linear_stats:
        return (intercept[0], beta[0][0])

    beta = round(beta[0][0], 5)
    if label_boolean:
        label = True if beta > 0 else False
    else:
        label = beta
    return [subset_x_var, subset_y_var, label]


def run_linear(data_x, data_y):
    lm = linear_model.LinearRegression()
    lm.fit(data_x, data_y)
    return lm.intercept_, lm.coef_


def drop_boolean_vars(df):
    #drop the booleans
    return df.drop(
        [
            'user_description_exists',
            'user_is_verified',
            'user_follower_count',
            # 'domain_redirect_count',
            'tweet_url_count',
            'user_following_count',
        ],
        axis=1)


# def drop_boolean_vars(df):
#     #drop the booleans
#     return df.drop(
#         [
#             'user_description_exists', 'user_is_verified',
#             'user_follower_count', 'user_lists_count', 'tweet_url_count'
#         ],
#         axis=1)
