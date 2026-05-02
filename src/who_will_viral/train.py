import os
import pickle
import re
import warnings

import optuna
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from who_will_viral.models.mlflow_utilities import run_experiment, setup_mlflow

load_dotenv()
RESULT_PATH = os.getenv("RESULT_PATH",'./reports/results/')
os.makedirs(RESULT_PATH, exist_ok=True)

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, train_path, val_path, test_path, cv=5):
        self.cv = cv
        setup_mlflow()
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        TARGET = 'is_trending'
        X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
        X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]
        X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        self.best_f1 = 0
        self.best_model_name = None

    def over_sample(self):
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = oversample.fit_resample(self.X_train, self.y_train)
        return X_over, y_over

    def under_sample(self):
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_under, y_under = undersample.fit_resample(self.X_train, self.y_train)
        return X_under, y_under

    def smote(self):
        smote = SMOTE(sampling_strategy='auto')
        X_smote, y_smote = smote.fit_resample(self.X_train, self.y_train)
        return X_smote, y_smote

    def tomek_links(self):
        tl = TomekLinks()
        X_tomek, y_tomek = tl.fit_resample(self.X_train, self.y_train)
        return X_tomek, y_tomek

    def get_sampled_data(self, sampling):
        if sampling == 'over':
            X_train, y_train = self.over_sample()
        elif sampling == 'under':
            X_train, y_train = self.under_sample()
        elif sampling == 'smote':
            X_train, y_train = self.smote()
        elif sampling == 'tomek':
            X_train, y_train = self.tomek_links()
        else:
            X_train, y_train = self.X_train, self.y_train
        return X_train, y_train

    def train_gaussian_nb(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)
        model = GaussianNB()
        param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,  # Use 5 fold on training set
            scoring='f1_macro',
        )
        metrics, model = run_experiment(f'Gaussian NB {sampling}', search, X_train, y_train, self.X_val, self.y_val)
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            self.best_model = model
            self.best_model_name = 'Gaussian NB'
        return search

    def train_knn(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)
        knn = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
        search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=self.cv,  # Use 5 fold on training set
            scoring='f1_macro',
        )
        metrics, model = run_experiment(
            f'Grid Search for KNN {sampling}', search, X_train, y_train, self.X_val, self.y_val
        )
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            self.best_model = model
            self.best_model_name = 'KNN'
        return search

    def train_ada(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)
        ada = AdaBoostClassifier()
        param_grid = {'n_estimators': [10, 50, 80, 100, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.5, 0.7, 1.0]}
        search = RandomizedSearchCV(
            estimator=ada,
            param_distributions=param_grid,
            cv=self.cv,  # Use 5 fold on training set
            scoring='f1_macro',
            n_iter=15,
        )
        metrics, model = run_experiment(
            f'Randomized Search for AdaBoost {sampling}', search, X_train, y_train, self.X_val, self.y_val
        )
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            self.best_model = model
            self.best_model_name = 'AdaBoost'
        return search

    def train_svc(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)
        svm_linear = LinearSVC(loss='squared_hinge', class_weight='balanced' if sampling == 'balanced' else None)
        param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}
        search = GridSearchCV(
            estimator=svm_linear,
            param_grid=param_grid,
            cv=self.cv,  # Use 5 fold on training set
            scoring='f1_macro',
        )
        metrics, model = run_experiment(
            f'Grid Search for LinearSVC with squared_hinge loss {sampling}',
            search,
            X_train,
            y_train,
            self.X_val,
            self.y_val,
        )
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            self.best_model = model
            self.best_model_name = 'LinearSVC'
        return search

    def train_random_forest(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=42, class_weight='balanced' if sampling == 'balanced' else None
            ),
            param_grid=param_grid,
            scoring='f1_macro',
            cv=self.cv,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        rf_metrics, rf_model = run_experiment(
            run_name=f'RF , GridSearch best {sampling}',
            model=best_model,
            X_ev=self.X_val,
            y_ev=self.y_val,
            X_tr=X_train,
            y_tr=y_train,
            params={**grid_search.best_params_, 'model': 'RandomForest', 'search': 'GridSearchCV'},
            tags={'stage': 'hyperparameter_tuning'},
            skip_fit=True,
        )
        if rf_metrics['f1'] > self.best_f1:
            self.best_f1 = rf_metrics['f1']
            self.best_model = rf_model
            self.best_model_name = 'RandomForest'
        return best_model

    def train_logistic_regression(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)
        model = LogisticRegression(max_iter=1000, class_weight='balanced' if sampling == 'balanced' else None)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
        }
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,  # Use 5 fold on training set
            scoring='f1_macro',
        )
        metrics, model = run_experiment(
            f'Grid Search for Logistic Regression {sampling}', search, X_train, y_train, self.X_val, self.y_val
        )
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            self.best_model = model
            self.best_model_name = 'Logistic Regression'
        return search

    def get_class_ratio(self, y):
        return (1 - sum(y) / len(y)) / (sum(y) / len(y))

    def train_XGBoost(self, sampling=None):
        sampling = '' if sampling is None else sampling
        X_train, y_train = self.get_sampled_data(sampling)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'scale_pos_weight': self.get_class_ratio(y_train),
                'random_state': 42,
                'n_jobs': -1,
            }
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(self.X_val)
            return f1_score(self.y_val, preds, average='macro')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_xgb = XGBClassifier(
            **study.best_params, scale_pos_weight=self.get_class_ratio(self.y_train), random_state=42
        )
        metrics, model = run_experiment(
            run_name=f'XGBoost, Optuna best, {sampling}',
            model=best_xgb,
            X_ev=self.X_val,
            y_ev=self.y_val,
            X_tr=X_train,
            y_tr=y_train,
            params={**study.best_params, 'model': 'XGBoost', 'search': 'Optuna'},
            tags={'stage': 'hyperparameter_tuning'},
        )

        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            self.best_model = model
            self.best_model_name = 'XGBoost'
        return best_xgb

    def get_test_report(self):
        if self.best_model is None:
            print('No model has been trained yet!')
            return None
        y_pred = self.best_model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(f'Test Report for Best Model: {self.best_model_name}')
        print(report)
        file_path = os.path.join(RESULT_PATH, f"{self.best_model_name}_report.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.best_model_name}\n\n")
            f.write(report)

        self.save_best_model()
        return report

    def save_best_model(self):
        if self.best_model is None:
            print('No model has been trained yet!')
            return None

        MODEL_PATH = os.getenv("MODEL_PATH")
        os.makedirs(MODEL_PATH, exist_ok=True)

        # Find the next available version number
        model_base_name = self.best_model_name.replace(" ", "_")
        existing = [
            f for f in os.listdir(MODEL_PATH)
            if re.match(rf"^{re.escape(model_base_name)}_v\d+\.pkl$", f)
        ]
        if existing:
            versions = [int(re.search(r'_v(\d+)\.pkl$', f).group(1)) for f in existing]
            next_version = max(versions) + 1
        else:
            next_version = 1

        filename = f"{model_base_name}_v{next_version}.pkl"
        file_path = os.path.join(MODEL_PATH, filename)

        with open(file_path, "wb") as f:
            pickle.dump(self.best_model, f)

        print(f"Best model saved to: {file_path}")
        return file_path


if __name__ == '__main__':
    if not os.getenv('CI'):
        trainer = ModelTrainer(
            train_path=os.getenv("SCALED_TRAIN_PATH","./data/youtube/scaled_train.csv"),
            val_path=os.getenv("SCALED_VAL_PATH",'./data/youtube/scaled_val.csv'),
            test_path=os.getenv("SCALED_TEST_PATH",'./data/youtube/scaled_test.csv'),
        )
        print('Starting training with different sampling techniques and models...')
        trainer.train_knn(sampling='smote')
        trainer.train_ada(sampling='smote')
        trainer.train_gaussian_nb(sampling='smote')
        trainer.train_svc(sampling='balanced')
        trainer.train_random_forest(sampling='balanced')
        trainer.train_logistic_regression(sampling='balanced')
        trainer.train_XGBoost(sampling='balanced')
        print(f'Best Model: {trainer.best_model_name} with macro F1 Score: {trainer.best_f1:.4f}')
        report = trainer.get_test_report()
