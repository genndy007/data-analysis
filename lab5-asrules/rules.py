from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from csv import reader


# Dataset: list of transactions
def csv_to_stacklist(file):
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        dataset = list(csv_reader)
    return dataset


def basket_bool_df(dst):
    te = TransactionEncoder()
    te_ary = te.fit(dst).transform(dst)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def main(file, support, confidence, lift):
    dst = csv_to_stacklist(path)
    df = basket_bool_df(dst)
    frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
    print(frequent_itemsets)
    res = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)[
        ['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    print(res[res['lift'] >= lift].to_string())


path = 'groceries.csv'
s = 0.02
c = 0.25
l = 1.7
if __name__ == '__main__':
    print('Dataset: \nhttps://www.kaggle.com/irfanasrullah/groceries?select=groceries.csv')
    main(path, s, c, l)
