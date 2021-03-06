from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from transformers import get_linear_schedule_with_warmup
from models.nn import HateSpeechClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import numpy as np
import nltk

import gc

from collections import defaultdict


def create_data_loader(data, tokenizer, max_len, batch_size):
    class HateSpeechDataset(Dataset):
        def __init__(self, comments, targets, tokenizer, max_len):
            self.comments = comments
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.comments)

        def __getitem__(self, item):
            comment = self.comments[item]
            target = self.targets[item]

            encoding = self.tokenizer.encode_plus(
                comment,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'comment_text': comment,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)
            }

    dataset = HateSpeechDataset(
        comments=data.text.to_numpy(),
        targets=data.toxicity_degree.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device, dtype=torch.long)
        attention_mask = d['attention_mask'].to(device, dtype=torch.long)
        targets = d['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        # To avoid the exploiding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        del input_ids, attention_mask, targets, outputs, preds, loss
        torch.cuda.empty_cache()
        gc.collect()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_nn(model, data_loader, device, n_examples):
    model = model.eval()
    all_preds = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device, dtype=torch.long)
            attention_mask = d['attention_mask'].to(device, dtype=torch.long)
            targets = d['targets'].to(device, dtype = torch.long)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.tolist())
            correct_predictions += torch.sum(preds == targets)

            del input_ids, attention_mask, outputs, preds
            torch.cuda.empty_cache()
            gc.collect()

    return all_preds, correct_predictions.double() / n_examples


def evaluate_model(original_data, model_config, eval_config, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    skf = StratifiedKFold(n_splits=eval_config['n_folds'])
    accuracies = []
    fold = 1

    data = original_data.copy()
    basic_cols = ['text']
    manual_cols = ['constructive', 'toxic', 'sarcasm_irony', 'mockery_ridicule', 'insults', 'argument_discussion',
                   'negative_toxic_lang', 'aggressiveness', 'intolerance']
    manual_transformation = {'sí': 1, 'si': 1, 'no': 0, 'd': 0.5}
    label_col = 'toxicity_degree'
    implemented_models = ['Random Forest', 'SVC', 'Logistic Regression']

    if eval_config['basic_manual_both'] == 2:
        for col in manual_cols:
            data[col] = data[col].str.lower().map(manual_transformation)
        data = data[basic_cols + manual_cols + [label_col]]
    elif eval_config['basic_manual_both'] == 1:
        for col in manual_cols:
            data[col] = data[col].str.lower().map(manual_transformation)
        data = data[manual_cols + [label_col]]
    else:
        data = data[basic_cols + [label_col]]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for train_index, test_index in skf.split(data, data[label_col].values):
        train = data.loc[train_index, :]
        test = data.loc[test_index, :]
        if model_config['name'].startswith('bert'):
            # Create data loaders based on the data split
            train_data_loader = create_data_loader(train, model_config['tokenizer'], model_config['max_len'],
                                                   model_config['batch_size'])
            test_data_loader = create_data_loader(test, model_config['tokenizer'], model_config['max_len'],
                                                  model_config['batch_size'])

            # Create the model and load it into the device
            model = HateSpeechClassifier(model_config['name'], data[label_col].nunique())
            model = model.to(device)

            # Add the Adam optimizer
            optimizer = torch.optim.Adam(params=model.parameters(), lr=model_config['learning_rate'])
            total_steps = len(train_data_loader) * model_config['epochs']

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            # Use the cross entropy loss
            weights = torch.Tensor(1/train.groupby('toxicity_degree').size().sort_index().values)
            loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)

            # Evaluate model on test
            history = defaultdict(list)

            for epoch in range(model_config['epochs']):

                print('Epoch {}/{}'.format(epoch + 1, model_config['epochs']))
                print('-' * 10)

                train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler,
                                                    len(train_index))

                print(f'Train loss {train_loss} accuracy {train_acc}')

                history['train_acc'].append(train_acc)
                history['train_loss'].append(train_loss)
                
                y_pred, test_acc = eval_nn(model, test_data_loader, device, len(test))

                print(f'Test  accuracy {test_acc}')
                print()

            # y_pred = eval_nn(model, test_data_loader, device, len(val))

        elif model_config['name'] in implemented_models:
            train_x = train[[c for c in train.columns if c != label_col]]
            train_y = train[label_col].values

            test_x = test[[c for c in test.columns if c != label_col]]

            if eval_config['basic_manual_both'] != 1:
                bow = TfidfVectorizer(strip_accents=model_config['strip_accents'], stop_words=model_config['stop_words'])
                train_bow_feats = bow.fit_transform(train_x.text.values).todense()

                # Perform dimensionality reduction
                pca = PCA(n_components=model_config['PCA_components'], svd_solver=model_config['svd_solver'])
                train_bow_feats = pca.fit_transform(train_bow_feats)
                test_bow_feats = bow.transform(test_x.text.values).todense()
                test_bow_feats = pca.transform(test_bow_feats)

                train_x.drop('text', axis=1, inplace=True)
                test_x.drop('text', axis=1, inplace=True)

                train_x = np.hstack((train_x.values, train_bow_feats))
                test_x = np.hstack((test_x.values, test_bow_feats))

            if model_config['name'] == 'Random Forest':
                rfc = RandomForestClassifier(n_estimators=model_config['n_trees'], criterion=model_config['criterion'],
                                             max_features=model_config['n_feats'], bootstrap=model_config['bootstrap'])
                rfc.fit(train_x, train_y)

                if eval_config['log']:
                    print(rfc.feature_importances_)

                y_pred = rfc.predict(test_x)

            elif model_config['name'] == 'SVC':
                svm = SVC(kernel=model_config['kernel'], decision_function_shape=model_config['decision_func'],
                          gamma=model_config['gamma'], C=model_config['penalty'])
                svm.fit(train_x, train_y)
                y_pred = svm.predict(test_x)

            elif model_config['name'] == 'Logistic Regression':
                lr = LogisticRegression(penalty=model_config['penalty'], solver=model_config['solver'],
                                        multi_class=model_config['multi_class'])
                lr.fit(train_x, train_y)
                y_pred = lr.predict(test_x)
        else:
            print('No valid model has been selected')
            return

        accuracies.append(f1_score(test[label_col].values, y_pred, labels=data[label_col].unique(), average='macro'))
        # print('Accuracy for Fold', fold, 'is:', np.round(accuracies[-1], 4))

        fold += 1

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # print('Total Prediction Accuracy is:', np.round(mean_accuracy, 4), '\u00B1', np.round(std_accuracy, 4))

    return mean_accuracy, std_accuracy
