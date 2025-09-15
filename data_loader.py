import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, test_ratio=0.2, seed=42, target_col="median_house_value"):
        self.path = kagglehub.dataset_download("camnugent/california-housing-prices")
        self.df = pd.read_csv(self.path+"/housing.csv")
        self.target_col = target_col
        self.test_ratio = test_ratio
        self.seed = seed
        np.random.seed(self.seed)

    def load_sklearn_format(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_ratio, random_state=self.seed)
        return X_train, y_train, X_test, y_test

    def load_custom_format(self):
        category_cols = self.df.select_dtypes(include=["object"]).columns

        features = pd.get_dummies(
            self.df.drop(columns=[self.target_col]),
            columns=category_cols,
            drop_first=True,
            dtype=float
        )

        y = self.df[self.target_col].astype(float)

        data = pd.concat([features, y], axis=1).dropna()

        # Splitting train/test

        idx = np.arange(len(data))
        random = np.random.default_rng(self.seed)
        random.shuffle(idx)

        n_test = int(len(idx) * self.test_ratio)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        test = data.iloc[test_idx]
        train = data.iloc[train_idx]

        X_train = train[features.columns].values.astype(float)
        y_train = train[self.target_col].values.astype(float)
        X_test = test[features.columns].values.astype(float)
        y_test = test[self.target_col].values.astype(float)

        return X_train, y_train, X_test, y_test