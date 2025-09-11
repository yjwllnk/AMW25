from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from amw25.util.utils import plot_parity
import pickle, joblib

class MLP_Regressor():
    def __init__(self, config, X_train, X_test, y_train, y_test, X_val = None, y_val = None):
        self.config = config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        model = self.init_model()
        self.model = model

    def save_data(self, filename='data.pkl'):
        print('saved data')
        data = {'X_train': self.X_train, 'X_test': self.X_test,
                'y_train': self.y_train, 'y_test': self.y_test,}
        with open(f"{self.config['model']['save']}/{filename}", 'wb') as f:
            pickle.dump(data, f)

    def init_model(self):
        print('initiating model')
        model = MLPRegressor(**self.config['model']['mlp_args'])
        return model

    def fit_model(self):
        print('training model')
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        print('testing model')
        y_test_pred = self.model.predict(self.X_test)
        self.y_test_pred = y_test_pred

    def eval_model(self):
        print('evaluating model')
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        plot_parity(self.y_test, self.y_test_pred, test_r2, self.config)

    def save_model(self, filename='mlp_joblib.pkl'):
        print('saving model')
        model = self.model
        joblib.dump(model, f"{self.config['model']['save']}/{filename}")

    def main(self):
        self.save_data()
        self.fit_model()
        self.test_model()
        self.eval_model()
        self.save_model()

