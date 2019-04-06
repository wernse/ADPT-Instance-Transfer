import pandas as pd

model_name = './models/rdmadict.xlsx'


#Input is a normalised df with only the
def train_rdma_model(X_train):
    col_means = X_train.mean()

    mean_df = pd.DataFrame([col_means])
    writer = pd.ExcelWriter(model_name)
    mean_df.to_excel(writer)
    writer.save()


def test_rdma_model(normalised_df):
    # writer = pd.ExcelWriter('./sample/clusters/rdma.xlsx')
    mean_df = pd.read_excel(model_name)
    col_means = mean_df.loc[0]
    # print("test_rdma_model:col_means", col_means)
    # print("test_rdma_model:normalised_df pre")
    # mean_df.to_excel(writer, 'mean_df')
    # normalised_df.to_excel(writer, 'rdma_pre')
    # print(normalised_df.head())
    normalised_df = abs(normalised_df - col_means) / len(normalised_df.columns)
    # print("test_rdma_model:normalised_df after")
    # normalised_df.to_excel(writer, 'rdma_post')
    # print(normalised_df.head())
    rdma_df = normalised_df.sum(axis=1)
    # rdma_df.to_excel(writer, 'rdma_df')
    # print("rdma_df.head()")
    # print(rdma_df.head())
    # writer.save()
    return rdma_df