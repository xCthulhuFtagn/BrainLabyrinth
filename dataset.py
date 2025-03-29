import polars as pl
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from tqdm.notebook import tqdm

from utils import TSMOTE
#============================================================
# Enhanced Dataset Class with Proper Encapsulation
#============================================================
class EEGDataset(Dataset):
    def __init__(self, source, max_length=2000):
        self.df = pl.read_parquet(source) if isinstance(source, str) else source
        if 'orig_marker' in self.df.columns:
            self.df = self.df.drop('orig_marker')
        
        self.df = self.df.with_columns([
            pl.col("marker")
            .cast(pl.Utf8)
            .str.replace_all("Left", "0")      # replace exact string "Left" with "0"
            .str.replace_all("Right", "1")     # replace exact string "Right" with "1"
            .cast(pl.Int32)                      # now cast the string "0"/"1" -> int
            .alias("marker"),
            
            pl.col("prev_marker")
            .cast(pl.Utf8)
            .str.replace_all("Left", "0")
            .str.replace_all("Right", "1")
            .cast(pl.Int32)
            .alias("prev_marker"),
        ])
        
        self.event_ids = self.df['event_id'].unique().to_list()
        self.max_length = max_length
        
        # Keep time for sorting but exclude from features
        self.feature_cols = [c for c in self.df.columns 
                           if c not in {'event_id', 'marker', 'time', 'prev_marker'}]
        # self.feature_cols = [      
        #     'FP1', 'FPZ', 'FP2', "AF7", 'AF3', 'AF4', "AF8", 
        #     'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        #     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 
        #     'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 
        #     'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 
        #     'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 
        #     'P6', 'P8', 'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 
        #     'PO8', 'O1', 'OZ', 'O2'
        # ]
        
        print("Precomputing samples...")
        self._precompute_samples()
        print("Computing class weights...")
        self._class_weights = self.compute_class_weights()
    
    @property
    def class_weights(self):
        # Expose the computed weights as a property.
        return self._class_weights 

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def _precompute_samples(self):
        self.samples = []
        for event_id in tqdm(self.event_ids, desc='precomputing_samples'):
            # Sort by time within each event!
            event_data = self.df.filter(pl.col("event_id") == event_id).sort("time")
            features = torch.tensor(
                event_data.select(self.feature_cols).to_numpy(),
                dtype=torch.float32
            )
            features = self._pad_sequence(features)
            
            label = event_data['marker'][0]
            self.samples.append((
                torch.tensor(label, dtype=torch.float32), 
                features
            ))
    
    def compute_class_weights(self):
        """
        Compute inverse frequency weights based on the 'marker' column.
        Assumes markers are "Stimulus/A" and "Stimulus/P".
        """
        # Get unique combinations of event_id and marker.
        unique_events = self.df.select(["event_id", "marker"]).unique()
        
        # Use value_counts on the "marker" column.
        counts_df = unique_events["marker"].value_counts()

        # We'll use 'values' if it exists, otherwise 'marker'.
        d = { (row.get("values") or row.get("marker")): row["count"] 
            for row in counts_df.to_dicts() }
        
        weight_L = 1.0 / d.get(0, 1)
        weight_R = 1.0 / d.get(1, 1)
        return {"Left": weight_L, "Right": weight_R}
   
    def split_dataset(self, ratios=(0.7, 0.15, 0.15), seed=None):
        """
        Splits the dataset into three EEGDataset instances for train, val, and test.
        This method shuffles the event_ids and then partitions them based on the given ratios.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Copy and shuffle the event_ids
        event_ids = self.event_ids.copy()
        np.random.shuffle(event_ids)
        total = len(event_ids)
        
        n_train = int(ratios[0] * total)
        n_val   = int(ratios[1] * total)
        
        train_ids = event_ids[:n_train]
        val_ids   = event_ids[n_train:n_train+n_val]
        test_ids  = event_ids[n_train+n_val:]
        
        # Filter self.df for the selected event_ids
        train_df = self.df.filter(pl.col("event_id").is_in(train_ids))
        val_df   = self.df.filter(pl.col("event_id").is_in(val_ids))
        test_df  = self.df.filter(pl.col("event_id").is_in(test_ids))
        
        # Create new EEGDataset instances using the filtered data
        train_set = EEGDataset(train_df, self.max_length)
        val_set   = EEGDataset(val_df, self.max_length)
        test_set  = EEGDataset(test_df, self.max_length)
        
        return train_set, val_set, test_set

    def _pad_sequence(self, tensor):
        # Pre-allocate tensor for maximum efficiency
        padded = torch.zeros((self.max_length, tensor.size(1)), dtype=tensor.dtype)
        length = min(tensor.size(0), self.max_length)
        padded[:length] = tensor[:length]
        return padded
    
    def rebalance_by_tsmote(self):
        """TSMOTE implementation for temporal EEG data"""
        # Extract time-ordered features as 3D array (samples, timesteps, features)
        X = np.stack([features.numpy() for _, features in self.samples])
        y = np.array([label.item() for label, _ in self.samples])
        
        # Apply TSMOTE with temporal awareness
        
        
        # Find the index of 'prev_marker' in the feature columns
        prev_marker_idx = self.feature_cols.index('prev_marker')
        
        # Apply TSMOTE with the correct boolean column index
        tsmote = TSMOTE(bool_cols=[prev_marker_idx])
        X_res, y_res = tsmote.fit_resample(X, y)

        # Generate synthetic temporal events
        new_events = []
        new_event_id = self.df['event_id'].max() + 1
        time_base = np.arange(self.max_length)
        original_schema = self.df.schema

        # Create dtype conversion map
        dtype_map = {
            pl.Float64: np.float64,
            pl.Float32: np.float32,
            pl.Int64: np.int64,
            pl.Int32: np.int32,
            pl.Utf8: str,
        }

        # Process synthetic samples (original samples come first in X_res)
        for features_3d, label in zip(X_res[len(self.samples):], y_res[len(self.samples):]):
            event_data = {}

            # Ensure columns are added in the original DataFrame's order
            for col in self.df.columns:
                if col == 'event_id':
                    event_data[col] = [new_event_id] * self.max_length
                elif col == 'marker':
                    event_data[col] = [int(label)] * self.max_length  # Ensure label is integer
                elif col == 'time':
                    event_data[col] = time_base.copy().astype(np.int32)  # Match original time type
                else:
                    # Feature columns (excluding event_id, marker, time)
                    if col not in self.feature_cols:
                        continue  # Shouldn't happen as feature_cols covers all else
                    col_idx = self.feature_cols.index(col)
                    col_data = features_3d[:, col_idx]
                    schema_type = original_schema[col]

                    # Handle data types
                    if isinstance(schema_type, pl.List):
                        base_type = schema_type.inner
                        target_type = dtype_map.get(type(base_type), np.float64)
                    else:
                        target_type = dtype_map.get(type(schema_type), np.float64)
                    
                    col_data = col_data.astype(target_type)
                    
                    # Maintain integer precision for Int columns (e.g., prev_marker)
                    if schema_type in (pl.Int64, pl.Int32):
                        col_data = np.round(col_data).astype(int)
                    
                    event_data[col] = col_data

            # Create DataFrame with strict schema adherence
            event_df = pl.DataFrame(event_data).cast(original_schema)
            new_events.append(event_df)
            new_event_id += 1

        # Update dataset with synthetic temporal events
        self.df = pl.concat([self.df, *new_events])
        self.event_ids = self.df['event_id'].unique().to_list()
        self._precompute_samples()
        self._class_weights = self.compute_class_weights()
        return self

#============================================================
# Enhanced Dataset Class For EEGPT
#============================================================

class EEGPTDataset(Dataset):
    def __init__(self, source, max_length=1024, for_train=True, scaler=None):
        # Load the data from a file if source is a path.
        self.df = pl.read_parquet(source) if isinstance(source, str) else source
        self.for_train = for_train
        self.max_length = max_length

        # Preprocess markers and drop unwanted columns.
        if 'orig_marker' in self.df.columns:
            self.df = self.df.drop('orig_marker')
        self.df = self.df.with_columns([
            pl.col("marker")
              .cast(pl.Utf8)
              .str.replace_all("Left", "0")
              .str.replace_all("Right", "1")
              .cast(pl.Int32)
              .alias("marker"),
            pl.col("prev_marker")
              .cast(pl.Utf8)
              .str.replace_all("Left", "0")
              .str.replace_all("Right", "1")
              .cast(pl.Int32)
              .alias("prev_marker"),
        ])

        # Define feature columns (the list of channels, for example)
        self.feature_cols = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 
            'F2', 'F4', 'F6', 'F8', 'FT7', 
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 
            'C3', 'C1', 'CZ', 'C2', 'C4', 
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 
            'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 
            'TP8', 'P7', 'P5', 'P3', 'P1', 
            'PZ', 'P2', 'P4', 'P6', 'P8', 
            'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
            'O1', 'OZ', 'O2' 
        ]


        # Use a private attribute to store the scaler.
        self._scaler = scaler  # If provided externally, otherwise will be fitted in _precompute_samples

        # Get unique event ids (used for splitting and sample extraction)
        self.event_ids = self.df['event_id'].unique().to_list()

        print("Precomputing samples...")
        self._precompute_samples()
        print("Computing class weights...")
        self._class_weights = self.compute_class_weights()
   
    @property
    def class_weights(self):
        # Expose the computed weights as a property.
        return self._class_weights

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _pad_sequence(self, tensor):
        padded = torch.zeros((self.max_length, tensor.size(1)), dtype=tensor.dtype)
        length = min(tensor.size(0), self.max_length)
        padded[:length] = tensor[:length]
        return padded

    def _precompute_samples(self):
        # If training and no scaler is provided, fit one on all feature data.
        if self._scaler is None and self.for_train:
            X_full = self.df.select(self.feature_cols).to_numpy()
            self._scaler = MinMaxScaler().fit(X_full)
        
        self.samples = []
        for event_id in tqdm(self.event_ids, desc='Precomputing Samples'):
            # Sort event data by time.
            event_data = self.df.filter(pl.col("event_id") == event_id).sort("time")
            X_event = event_data.select(self.feature_cols).to_numpy()
            # Apply scaling if a scaler is available.
            if self._scaler is not None:
                X_event = self._scaler.transform(X_event)
            features = torch.tensor(X_event, dtype=torch.float32)
            features = self._pad_sequence(features)
            label = event_data['marker'][0]
            self.samples.append((torch.tensor(label, dtype=torch.float32), features))

    def compute_class_weights(self):
        unique_events = self.df.select(["event_id", "marker"]).unique()
        counts_df = unique_events["marker"].value_counts()
        d = { (row.get("values") or row.get("marker")): row["count"]
              for row in counts_df.to_dicts() }
        weight_0 = 1.0 / d.get(0, 1)
        weight_1 = 1.0 / d.get(1, 1)
        return {"Left": weight_0, "Right": weight_1}

    def split_dataset(self, ratios=(0.7, 0.15, 0.15), seed=None):
        if seed is not None:
            np.random.seed(seed)
        event_ids = self.event_ids.copy()
        np.random.shuffle(event_ids)
        total = len(event_ids)
        n_train = int(ratios[0] * total)
        n_val = int(ratios[1] * total)

        train_ids = event_ids[:n_train]
        val_ids = event_ids[n_train:n_train+n_val]
        test_ids = event_ids[n_train+n_val:]

        train_df = self.df.filter(pl.col("event_id").is_in(train_ids))
        val_df = self.df.filter(pl.col("event_id").is_in(val_ids))
        test_df = self.df.filter(pl.col("event_id").is_in(test_ids))

        # For validation and test sets, pass the scaler fitted on the training set.
        train_set = EEGPTDataset(train_df, self.max_length, for_train=True)
        val_set = EEGPTDataset(val_df, self.max_length, for_train=False, scaler=train_set._scaler)
        test_set = EEGPTDataset(test_df, self.max_length, for_train=False, scaler=train_set._scaler)
        return train_set, val_set, test_set

    def rebalance_by_tsmote(self):
        """TSMOTE implementation for temporal EEG data"""
        # Extract time-ordered features as 3D array (samples, timesteps, features)
        X = np.stack([features.numpy() for _, features in self.samples])
        y = np.array([label.item() for label, _ in self.samples])

        # Apply TSMOTE with temporal awareness
        # Find the index of 'prev_marker' in the feature columns
        prev_marker_idx = self.feature_cols.index('prev_marker')
        # Apply TSMOTE with the correct boolean column index
        tsmote = TSMOTE(bool_cols=[prev_marker_idx])
        X_res, y_res = tsmote.fit_resample(X, y)

        # Generate synthetic temporal events
        new_events = []
        new_event_id = self.df['event_id'].max() + 1
        time_base = np.arange(self.max_length)
        original_schema = self.df.schema

        # Create dtype conversion map
        dtype_map = {
            pl.Float64: np.float64,
            pl.Float32: np.float32,
            pl.Int64: np.int64,
            pl.Int32: np.int32,
            pl.Utf8: str,
        }

        # Process synthetic samples (original samples come first in X_res)
        for features_3d, label in zip(X_res[len(self.samples):], y_res[len(self.samples):]):
            event_data = {}

            # Ensure columns are added in the original DataFrame's order
            for col in self.df.columns:
                if col == 'event_id':
                    event_data[col] = [new_event_id] * self.max_length
                elif col == 'marker':
                    event_data[col] = [int(label)] * self.max_length  # Ensure label is integer
                elif col == 'time':
                    event_data[col] = time_base.copy().astype(np.int32)  # Match original time type
                else:
                    # Feature columns (excluding event_id, marker, time)
                    if col not in self.feature_cols:
                        continue  # Shouldn't happen as feature_cols covers all else
                    col_idx = self.feature_cols.index(col)
                    col_data = features_3d[:, col_idx]
                    schema_type = original_schema[col]

                    # Handle data types
                    if isinstance(schema_type, pl.List):
                        base_type = schema_type.inner
                        target_type = dtype_map.get(type(base_type), np.float64)
                    else:
                        target_type = dtype_map.get(type(schema_type), np.float64)

                    col_data = col_data.astype(target_type)

                    # Maintain integer precision for Int columns (e.g., prev_marker)
                    if schema_type in (pl.Int64, pl.Int32):
                        col_data = np.round(col_data).astype(int)

                    event_data[col] = col_data

            # Create DataFrame with strict schema adherence
            event_df = pl.DataFrame(event_data).cast(original_schema)
            new_events.append(event_df)
            new_event_id += 1

        # Update dataset with synthetic temporal events
        self.df = pl.concat([self.df, *new_events])
        self.event_ids = self.df['event_id'].unique().to_list()
        self._precompute_samples()
        self._class_weights = self.compute_class_weights()

        return self