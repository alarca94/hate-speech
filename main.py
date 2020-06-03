import os

import numpy as np
import pandas as pd

from models.decisionTree import DecisionTree
from models.PRISM_Algorithm import PRISM_Algorithm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def preprocess(data, columns):
    data['ToxicityDegree'] = data[['ToxicityDegree']].fillna(1)
    data = data.drop(['Toxic'], axis=1)
    columns.pop(columns.index('Toxic'))

    data.dropna(inplace=True)

    data.Insults = data.Insults.str.upper()
    data.Constructive = data.Constructive.str.upper()
    data['Sarcasm/Irony'].replace({'SI': 'SÍ'}, inplace=True)

    return data, columns


def evaluate_models(model, x_data, y_data, eval_type=0, attr_names=None, attr_vals=None):
    skf = StratifiedKFold(n_splits=5)
    accuracies = []
    fold = 1

    if eval_type in [0, 2] and model in [2, 3, 4, 5]:
        x_data = pd.DataFrame(x_data).replace({'SÍ': 1, 'NO': 0}).values

    if eval_type in [1, 2] and model in [0, 1]:
        return

    # if model != 5:
    #     continue

    for train_index, test_index in skf.split(x_data, y_data):
        if eval_type == 0:
            trn_x = x_data[train_index, :]
            trn_y = y_data[train_index]
            tst_x = x_data[test_index, :]
            tst_y = y_data[test_index]
        elif eval_type == 1:
            trn_x = x_data[train_index]
            trn_y = y_data[train_index]
            tst_x = x_data[test_index]
            tst_y = y_data[test_index]

            bow = TfidfVectorizer()
            trn_x = bow.fit_transform(trn_x).todense()

            pca = PCA(n_components=0.8, svd_solver='full')
            trn_x = pca.fit_transform(trn_x)

            tst_x = bow.transform(tst_x).todense()
            tst_x = pca.transform(tst_x)
        else:
            trn_x = x_data[train_index, :]
            trn_y = y_data[train_index]
            tst_x = x_data[test_index, :]
            tst_y = y_data[test_index]

            bow = TfidfVectorizer()  # max_features=300)
            trn_bow_feats = bow.fit_transform(trn_x[:, 0]).todense()
            pca = PCA(n_components=0.7, svd_solver='full')
            trn_bow_feats = pca.fit_transform(trn_bow_feats)

            tst_bow_feats = bow.transform(tst_x[:, 0]).todense()
            tst_bow_feats = pca.transform(tst_bow_feats)

            trn_x = np.hstack((trn_x[:, 1:], trn_bow_feats))
            tst_x = np.hstack((tst_x[:, 1:], tst_bow_feats))

        if model == 0:
            # Build Decision Tree model
            dt = DecisionTree(trn_x, trn_y, tree_type='C4.5')
            y_pred = [dt.predict(tst_x[i, :]) for i in range(tst_x.shape[0])]

            # dt.print_tree([c for c in manual_feats if c != y_label])
        elif model == 1:
            # Build PRISM rules model
            prism = PRISM_Algorithm(attr_names, attr_vals)
            prism.fit(trn_x, trn_y)
            y_pred = prism.predict(tst_x)

        elif model == 2:
            rfc = RandomForestClassifier(n_estimators=150, criterion='gini', max_features=3, bootstrap=True)
            rfc.fit(trn_x, trn_y)
            if eval_type == 0:
                print(rfc.feature_importances_)
            y_pred = rfc.predict(tst_x)

        elif model == 3:
            mlp = MLPClassifier(hidden_layer_sizes=(30, 30), activation='tanh', solver='adam', batch_size=32,
                                learning_rate='invscaling', learning_rate_init=0.01)
            mlp.fit(trn_x, trn_y)
            y_pred = mlp.predict(tst_x)

        elif model == 4:
            svm = SVC(kernel='linear', decision_function_shape='ovr', gamma='auto', C=100)
            svm.fit(trn_x, trn_y)
            y_pred = svm.predict(tst_x)

        elif model == 5:
            lr = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='auto')
            lr.fit(trn_x, trn_y)
            y_pred = lr.predict(tst_x)

        accuracies.append(accuracy_score(tst_y, y_pred))
        print('Accuracy for Fold', fold, 'is:', np.round(accuracies[-1], 4))

        fold += 1

    print('Total Prediction Accuracy is:', np.round(np.mean(accuracies), 4), '\u00B1', np.round(np.std(accuracies), 4))


def main():
    data_path = './data'
    data_file = 'fase1_20112019_2.xlsx'
    sheets = ['ECONOMÍA', 'INMIGRACIÓN', 'POLÍTICA', 'RELIGIÓN', ]
    models = {0: 'Decision Tree', 1: 'PRISM Rules', 2: 'Random Forest', 3: 'MLP Network', 4: 'SVM',
              5: 'Logistic Regression'}
    # model = 1  # 0: Decision Tree --- 1: PRISM Rules ---

    eval_message = 'Introduce the evaluation type of the models:\n1) Manual Features\n2) Bag-Of-Words Features\n' + \
        '3) BOW and Manual Features\n'
    eval_type = int(input(eval_message)) - 1

    np.random.seed(5)

    useful_columns = ['ID', 'Column1', 'COM. CONSTRUCTIU', 'COM. TÒXIC', 'GRAU TOXICITAT', 'Sarcasme/Ironia',
                      'Burla/ridiculització', 'Insults', 'Argumentació/Diàleg', 'Llenguatge negatiu/tòxic']

    columns = ['TextData', 'Constructive', 'Toxic', 'ToxicityDegree', 'Sarcasm/Irony',
               'Mockery/Ridicule', 'Insults', 'Argument/Discussion', 'NegativeToxicLanguage']

    data = pd.DataFrame()
    for sheet in sheets:
        # Read data
        new_data = pd.read_excel(os.path.join(data_path, data_file), sheet_name=sheet, index_col='ID',
                                 usecols=useful_columns)
        # Rename the columns
        new_data.columns = columns

        data = pd.concat([data, new_data], ignore_index=True)

    print('\n---------------------\nSummary of the dataset BEFORE preprocessing:\n')
    print(data.info())

    # PREPROCESS THE DATA
    data, columns = preprocess(data, columns)

    y_label = 'ToxicityDegree'

    print('\n---------------------\nDistinct values of each feature:\n')
    for column in data.columns:
        if column != 'TextData':
            print(column, data[column].unique())

    print('\n---------------------\nSummary of the dataset AFTER preprocessing:\n')
    print(data.info())

    x_data = None
    y_data = data[y_label].values

    # TEST with different type of data
    manual_feats = columns[1:]
    if eval_type == 0:
        manual_data = data[manual_feats]

        x_data = manual_data[[c for c in manual_feats if c != y_label]].values

        attr_names = [c for c in manual_feats if c != y_label]
        attr_vals = {}
        for i in range(x_data.shape[1]):
            attr_vals[attr_names[i]] = np.unique(x_data[:, i])
    else:
        attr_names = None
        attr_vals = None

        if eval_type == 1:
            x_data = data.TextData.apply(lambda x: np.str_(x)).values
        else:
            x_data = np.hstack((data.TextData.apply(lambda x: np.str_(x)).values.reshape(-1, 1),
                                data[[c for c in manual_feats if c != y_label]].values))

    for model in models.keys():
        print('\n---------------------\nStarting Training and prediction with', models[model], ':\n')
        evaluate_models(model, x_data, y_data, eval_type=eval_type, attr_names=attr_names, attr_vals=attr_vals)

    # TEST with bag-of-words


    # print('Finished')


if __name__ == '__main__':
    main()
