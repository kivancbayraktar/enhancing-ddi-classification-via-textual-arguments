from typing import List, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedKFold, train_test_split


class DatasetSplitter(BaseModel):
    fold_size: int = Field(default=5, ge=2)
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    shuffle: bool = True
    random_state: int = Field(default=42)

    class Config:
        arbitrary_types_allowed = True

    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
            pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index, List[np.ndarray], List[np.ndarray]]:
        print(
            f"Splitting dataset into {self.fold_size} folds with shuffle={self.shuffle}...")
        # TODO check it
        if len(y.shape) == 1:
            y = pd.Series(np.expand_dims(y.to_numpy(), axis=1).flatten())
        stacked = np.vstack(tuple(y.to_numpy()))
        stratify = np.argmax(stacked, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=self.shuffle, test_size=self.test_size, stratify=stratify)

        k_fold = StratifiedKFold(
            n_splits=self.fold_size, shuffle=self.shuffle, random_state=self.random_state)
        folds = k_fold.split(X_train, np.argmax(
            np.vstack(y_train.to_numpy()), axis=1))
        train_idx_arr = []
        val_idx_arr = []
        for i, (train_index, val_index) in enumerate(folds):
            train_idx_arr.append(train_index)
            val_idx_arr.append(val_index)

        return X_train, X_test, y_train, y_test, X_train.index, X_test.index, train_idx_arr, val_idx_arr
