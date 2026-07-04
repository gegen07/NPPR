import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TransactionPreprocessor:
    """Fit encoders and scalers on training data only, then transform any split."""

    def __init__(
        self,
        categorical_features: dict[str, int],
        numeric_mappings: dict[str, str] | None = None,
        time_gap_source: str = "timestamp_diff",
        time_features_zero_indexed: list[str] | None = None,
        preprocessed: bool = False,
    ):
        self.categorical_features = categorical_features
        self.numeric_mappings = numeric_mappings or {}
        self.time_gap_source = time_gap_source
        self.time_features_zero_indexed = time_features_zero_indexed or []
        self.preprocessed = preprocessed
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TransactionPreprocessor":
        df = self._apply_time_index_adjustments(df.copy())

        if self.preprocessed:
            for col in self.categorical_features:
                self._validate_preprocessed_categorical(df, col)
        else:
            for col in self.categorical_features:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.label_encoders[col] = encoder

        for output_col, source_col in self.numeric_mappings.items():
            scaler = StandardScaler()
            scaler.fit(self._source_values(df, source_col))
            self.scalers[output_col] = scaler

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call fit() on training data before transform().")

        df = self._apply_time_index_adjustments(df.copy())

        if self.preprocessed:
            for col in self.categorical_features:
                self._validate_preprocessed_categorical(df, col)
                df[col] = df[col].astype(int) + 1
        else:
            for col, encoder in self.label_encoders.items():
                known = set(encoder.classes_)
                values = df[col].astype(str)
                if not set(values.unique()).issubset(known):
                    unknown = set(values.unique()) - known
                    raise ValueError(
                        f"Column {col} contains unseen categories during transform: {unknown}"
                    )
                df[col] = encoder.transform(values) + 1

        df["Time_Gap"] = self._raw_time_gaps(df)

        for output_col, source_col in self.numeric_mappings.items():
            df[output_col] = self.scalers[output_col].transform(
                self._source_values(df, source_col)
            )

        return df

    def _apply_time_index_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.time_features_zero_indexed:
            if col in df.columns:
                df[col] = df[col] - 1
        return df

    def _validate_preprocessed_categorical(self, df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: {col}")

        values = df[col].astype(int)
        min_val = int(values.min())
        max_val = int(values.max())
        num_categories = self.categorical_features[col]

        if min_val < 0 or max_val >= num_categories:
            raise ValueError(
                f"Column {col} must use 0-indexed category ids in "
                f"[0, {num_categories - 1}], got [{min_val}, {max_val}]"
            )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _source_values(self, df: pd.DataFrame, source_col: str) -> pd.DataFrame:
        if source_col in (self.time_gap_source, "Time_Gap"):
            return self._raw_time_gaps(df).to_frame(name=source_col)
        return df[[source_col]].astype(float)

    @staticmethod
    def _raw_time_gaps(df: pd.DataFrame) -> pd.Series:
        if "Time_Gap" in df.columns:
            return df["Time_Gap"].fillna(0).replace(0, 1e-6)
        return df["timestamp_diff"].fillna(0).replace(0, 1e-6)
