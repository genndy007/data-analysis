from sklearn import datasets
from dtreeviz.trees import *

# C4.5


def main(dataset, predict_sample):
    print('\nDetermine class for:')
    print(*[f"{dataset['feature_names'][i]}: {predict_sample[i]}" for i in range(len(predict_sample))], sep='\n')
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(dataset.data, dataset.target)
    print('\nTree parameters:', *list(clf.get_params().items()), sep='\n')
    print('Predicted class for the sample is', *clf.predict([predict_sample]))
    viz = dtreeviz(clf, dataset.data, dataset.target, target_name='wine', feature_names=dataset.feature_names,
                   class_names=list(dataset.target_names))
    viz.save('res.svg')


test = [11.62, 1.99, 2.28, 18, 98, 3.02,
        2.26, .17, 1.35, 3.25, 1.16, 2.96, 345]
dst = datasets.load_wine()
if __name__ == '__main__':
    print('Dataset: \nhttps://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
    main(dst, test)
    print('Tree visualisation saved as "res.svg".')
