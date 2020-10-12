import pandas as pd
import matplotlib.pyplot as plt


def data_dist(data, bins=50, path=False):
    """display a distribution of the features and labels assumes labels to be ['theta', 'phi', 'lam', 'E']
    when path=False the figures wont be saved"""
    feat_cols = ['theta', 'phi', 'lam', 'E']
    # the labels are supposed to be everything else so:
    label_cols = [x for x in data.columns if x not in feat_cols]
    # plot phi theta and lam
    features = plt.figure(figsize=(6.4 * len(feat_cols), 4.8))
    features.suptitle('features distribution')
    for idx, col in enumerate(feat_cols):
        plt.subplot(1, len(feat_cols), idx + 1)
        plt.hist(data[col], bins=bins)
        plt.title(f"{col} dist")
    features.show()
    # plot the labels distributions
    labels = plt.figure(figsize=(6.8 * len(label_cols), 4.8))
    labels.suptitle('labels distribution')
    for idx, col in enumerate(label_cols):
        plt.subplot(1, len(label_cols), idx + 1)
        plt.hist(data[col], bins=bins)
        plt.title(f"{col} dist")
    labels.show()
    # save figs
    if path:
        features.savefig(path + "/features.png", format='png')
        labels.savefig(path + "/labels.png", format='png')
    return 1


if __name__ == "__main__":
    data = pd.read_csv(r'../datasets/universal_error/V1/U3_6_6.csv', index_col=[0])
    data_dist(data, path="../graphics/U3_6_6_graphics")
