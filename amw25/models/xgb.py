from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from amw25.util.utils import plot_parity
import pickle, joblib
import pandas as pd
from abc import abstractmethod

class StopWhenReachedScore:
    def __init__(self, score, config):
        self.score = score
        self.dir = config['model']['optuna']['direction']

    def __call__(self, study, trial) -> None:
        if self.dir == 'minimize':
            if trial.user_attrs.get('score', float('inf')) <= self.score:
                print(f"Score {trial.user_attrs['score']} reached the target {self.score}.")
                study.stop()
        else:
            if trial.user_attrs.get('score', 0) >= self.score:
                print(f"Score {trial.user_attrs['score']} reached the target {self.score}.")
                study.stop()

class XGB_Regressor():
    def __init__(self, config, X_train, X_test, y_train, y_test, X_val=None, y_val=None):
        self.config = config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.targets = self.config['data']['target']
        print(self.targets)

    @abstractmethod
    def save_data(self, filename='data.pkl'):
        print('saved data')
        data = {'X_train': self.X_train, 'X_test': self.X_test,
                'y_train': self.y_train, 'y_test': self.y_test,}
        with open(f"{self.config['model']['save']}/{filename}", 'wb') as f:
            pickle.dump(data, f)

    def get_mae(self, y_true, y_pred):
        mae1 = mean_absolute_error(y_true[self.targets[0]], y_pred[self.targets[0]])
        mae2 = mean_absolute_error(y_true[self.targets[1]], y_pred[self.targets[1]])
        mae3 = mean_absolute_error(y_true[self.targets[2]], y_pred[self.targets[2]])
        return [mae1, mae2, mae3]

    def get_r2(self, y_true, y_pred):
        print(y_true)
        print(y_pred)
        r1 = r2_score(y_true[self.targets[0]], y_pred[0])
        r2 = r2_score(y_true[self.targets[1]], y_pred[self.targets[1]])
        r3 = r2_score(y_true[self.targets[2]], y_pred[self.targets[2]])
        return [r1, r2, r3]

    def get_rmse(self, y_true, y_pred):
        rmse1 = root_mean_squared_error(y_true[self.targets[0]], y_pred[self.targets[0]])
        rmse2 = root_mean_squared_error(y_true[self.targets[1]], y_pred[self.targets[1]])
        rmse3 = root_mean_squared_error(y_true[self.targets[2]], y_pred[self.targets[2]])
        return [rmse1, rmse2, rmse3]


    def r2_scorer(self, y_true, y_pred):
        r1, r2, r3 = self.get_r2(y_true, y_pred)
        r_mean = (r1 + r2 + r3) / 3
        return r_mean

    def mae_scorer(self, y_true, y_pred):
        mae1, mae2, mae3 = self.get_mae(y_true, y_pred)
        mae_mean = (mae1 + mae2 + mae3) / 3
        return mae_mean

    def rmse_scorer(self, y_true, y_pred):
        rmse1, rmse2, rmse3 = self.get_rmse(y_true, y_pred)
        rmse_mean = (rmse1 + rmse2 + rmse3)/3
        return rmse_mean

    def study_objective(self, trial):
        print('initiating model')
        conf = self.config['model']['xgb_args']
        xgb_args = {'n_estimators': trial.suggest_int('n_estimators', conf['n_estim'][0], conf['n_estim'][1], step=conf['n_estim'][-1]),
                    'max_depth': trial.suggest_int('max_depth', conf['max_depth'][0], conf['max_depth'][1], step=conf['max_depth'][-1]),
                    'learning_rate': trial.suggest_float('learning_rate', conf['lr'][0], conf['lr'][1], log=conf['lr'][-1]),
                    'reg_alpha': trial.suggest_float('reg_alpha', conf['alpha'][0], conf['alpha'][1]),
                    'reg_lambda': trial.suggest_float('reg_lambda', conf['lambda'][0], conf['lambda'][1]),
                    'early_stopping_rounds': conf['early_stopping'],
                    }
        model = XGBRegressor(**xgb_args)
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
        y_pred = model.predict(self.X_test)
        if self.config['model']['optuna']['score'] == 'r2':
            score = self.r2_scorer(self.y_test, y_pred)
        elif self.config['model']['optuna']['score'] == 'mae':
            score = self.mae_scorer(self.y_test, y_pred)
        elif self.config['model']['optuna']['score'] == 'rmse':
            score = self.rmse_scorer(self.y_test, y_pred)
        trial.set_user_attr('score', score)
        return score

    def test_study_result(self):
        model = XGBRegressor(**self.best_params)
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
        y_pred_train = model.predict(self.X_train)
        y_pred_val = model.predict(self.X_val)

        r2_train = self.get_r2(self.y_train, y_pred_train)
        r2_test = self.get_r2(self.y_test, y_pred_test)
        print(f"R2 Train: {r2_train}, R2 Test: {r2_test}")
        print(f"MAE Train: {self.get_mae(self.y_train, y_pred_train)}, MAE Test: {self.get_mae(self.y_test, y_pred_test)}")

        self.best_model = model

        data = {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'y_pred_train': y_pred_train,
            # 'y_pred_test': y_pred_test,
            'y_pred_val': y_pred_val,
            'params': self.best_params,
            'study': self.study,
        }

        with open(f"{self.config['model']['save']}/xgb_data.pkl", 'wb') as f:
            pickle.dump(data, f)

        r2s = self.get_r2(self.y_val, y_pred_val)
        maes = self.get_mae(self.y_val, y_pred_val)
        rmses = self.get_rmse(self.y_val, y_pred_val)

        plot_args = {
            'y_test': self.y_val,
            'y_pred': y_pred_val,
            'r2': r2s,
            'mae': maes,
            'rmse': rmses,
            'config': self.config,
        }

        plot_parity(**plot_args)

    def study_model(self):
        stop_search = StopWhenReachedScore(self.config['model']['optuna']['stop'], self.config)
        study = optuna.create_study(direction=self.config['model']['optuna']['direction'], sampler=optuna.samplers.TPESampler(seed=self.config['model']['optuna']['seed']))
        study.optimize(self.study_objective, callbacks=[stop_search])
        print(f"Best parameters: {study.best_params}")
        self.study = study
        self.best_params = study.best_params

    def save_model(self, filename='xgb_joblib.pkl'):
        print('saving model')
        model = self.best_model
        joblib.dump(model, f"{self.config['model']['save']}/{filename}")

    def main(self):
        self.study_model()
        self.test_study_result()
        self.save_model()

