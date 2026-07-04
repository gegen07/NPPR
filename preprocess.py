import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TransactionPreprocessor:
    """Fit encoders and scalers on training data only, then transform any split."""

    def __init__(
        self,
        categorical_features: dict[str, int],
        numeric_mappings: dict[str, str],
        time_gap_source: str = "timestamp_diff",
        time_features_zero_indexed: list[str] | None = None,
    ):
        self.categorical_features = categorical_features
        self.numeric_mappings = numeric_mappings
        self.time_gap_source = time_gap_source
        self.time_features_zero_indexed = time_features_zero_indexed or []
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TransactionPreprocessor":
        df = df.copy()

        for col in self.time_features_zero_indexed:
            if col in df.columns:
                df[col] = df[col] - 1

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

        df = df.copy()

        for col in self.time_features_zero_indexed:
            if col in df.columns:
                df[col] = df[col] - 1

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
