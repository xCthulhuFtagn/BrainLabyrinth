import polars as pl
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm.notebook import tqdm

from utils import TSMOTE


#============================================================
# Enhanced Dataset Class with Proper Encapsulation
#============================================================
class EEGDatasetV2(Dataset):
    def __init__(self, source, max_length=2000):
        self.df = pl.read_parquet(source) if isinstance(source, str) else source
        # if 'orig_marker' in self.df.columns:
        #     self.df = self.df.drop('orig_marker')
        
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
            
            pl.col("prev_prev_marker")
            .cast(pl.Utf8)
            .str.replace_all("Left", "0")
            .str.replace_all("Right", "1")
            .cast(pl.Int32)
            .alias("prev_prev_marker"),
        ])
        
        self.event_ids = self.df['event_id'].unique().to_list()
        self.max_length = max_length
        # Keep time for sorting but exclude from features
        self.feature_cols = [c for c in self.df.columns 
                           if c not in {'event_id', 'marker', 'time', 'orig_marker'}]
        print(self.feature_cols)
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

            original_label = event_data['orig_marker'][0] == 'Stimulus/P'
            self.samples.append((
                torch.tensor(label, dtype=torch.float32), 
                features,
                torch.tensor(original_label, dtype=torch.float32)
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
        train_set = EEGDatasetV2(train_df, self.max_length)
        val_set   = EEGDatasetV2(val_df, self.max_length)
        test_set  = EEGDatasetV2(test_df, self.max_length)
        
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


class V2EEGDataset(Dataset):
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
                           if c not in {'event_id', 'marker', 'time'}]
        
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
        Compute the raw counts of unique events for each class ('marker' column)
        within this dataset split.
        """
        # Get unique combinations of event_id and marker for this dataset part.
        unique_events = self.df.select(["event_id", "marker"]).unique()

        # Use value_counts on the "marker" column.
        counts_df = unique_events["marker"].value_counts()

        # Create a dictionary of counts {marker_value: count}
        # Handle cases where a class might be missing after splitting/filtering
        counts_dict = {
            row.get("marker"): row["count"]
            for row in counts_df.to_dicts()
            if row.get("marker") in [0, 1] # Only consider valid markers 0 and 1
        }

        # Get counts for Left (0) and Right (1), defaulting to 0 if absent
        count_L = counts_dict.get(0, 0)
        count_R = counts_dict.get(1, 0)

        # --- RETURN THE COUNTS ---
        # Returning counts is generally more flexible than returning the ratio directly
        return {"Left": count_L, "Right": count_R}
   
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