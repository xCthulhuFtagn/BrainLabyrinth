import numpy as np
from numba import njit

from tslearn.neighbors import KNeighborsTimeSeries

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate function for variable-length EEG feature sequences.

    Each sample is expected to be a tuple (label, feature), where:
    - label is a scalar tensor (or 1D tensor) representing the class/target.
    - feature is a tensor of shape (seq_len, num_channels), where seq_len may vary.

    This function stacks labels and pads features along the time dimension so that
    all sequences in the batch have the same length.
    """
    # Unzip the batch into labels and features
    labels, features = zip(*batch)
    
    labels = torch.stack(labels)
    padded_features = pad_sequence(features, batch_first=True)
    
    return labels, padded_features

@njit(fastmath=True)
def fast_interpolate(original, neighbor, alpha):
    """Numba-accelerated linear interpolation for numeric columns."""
    return (1 - alpha) * original + alpha * neighbor

class TSMOTE:
    def __init__(self, 
                 n_neighbors=3, 
                 time_slices=10, 
                 bool_cols=None):
        """
        :param n_neighbors: Number of neighbors for KNN
        :param time_slices: Number of slices to split each time series
        :param bool_cols:   List (or array) of indices for boolean columns
        """
        self.n_neighbors = n_neighbors
        self.time_slices = time_slices
        self.slice_size = None  # will be set after seeing data
        self.bool_cols = bool_cols if bool_cols is not None else []
        # numeric_cols will be determined at fit-time, after we see total channels.

    def _slice_time_series(self, X):
        """Split into time slices: (N, 2000, ch) -> (N, time_slices, slice_size, ch)."""
        return X.reshape(X.shape[0], self.time_slices, self.slice_size, X.shape[2])

    def _generate_synthetic(self, minority_samples, bool_probs):
        """
        Generate full-length synthetic samples.
        :param minority_samples: Array of shape (N_minority, 2000, ch)
        :param bool_probs:       Dict mapping boolean column index -> probability of 1
        """
        # slice_size was computed earlier in fit_resample.
        sliced_data = self._slice_time_series(minority_samples)  # shape (N, slices, slice_size, ch)
        syn_samples = []

        # We'll figure out numeric_cols from total channels
        all_cols = list(range(minority_samples.shape[2]))
        numeric_cols = [c for c in all_cols if c not in self.bool_cols]

        for sample_idx in tqdm(range(sliced_data.shape[0]), desc="Generating synthetic"):
            synthetic_slices = []

            # For each time slice
            for slice_idx in range(self.time_slices):
                # Split data into included (numeric) columns vs. excluded (boolean) columns
                slice_incl = sliced_data[:, slice_idx, :, :][:, :, numeric_cols]  # (N, slice_size, #numeric)
                slice_excl = sliced_data[:, slice_idx, :, :][:, :, self.bool_cols] # (N, slice_size, #bool)

                # Fit KNN on included (numeric) data only
                knn = KNeighborsTimeSeries(n_neighbors=self.n_neighbors, metric='dtw')
                knn.fit(slice_incl)  # each entry is shape (slice_size, #numeric)

                # The sample's numeric slice
                original_slice_incl = slice_incl[sample_idx]  # shape (slice_size, #numeric)

                # Find neighbors for this numeric slice
                neighbors = knn.kneighbors(original_slice_incl[np.newaxis], 
                                           return_distance=False)[0]
                neighbor_idx = np.random.choice(neighbors)

                neighbor_slice_incl = slice_incl[neighbor_idx]  # shape (slice_size, #numeric)

                # Interpolate for numeric columns
                alpha = np.random.uniform(0.2, 0.8)
                # Using fast_interpolate or direct calculation:
                synthetic_slice_incl = fast_interpolate(original_slice_incl, 
                                                        neighbor_slice_incl, 
                                                        alpha)

                # For boolean columns: sample from distribution
                # We'll create an array of shape (slice_size, #bool)
                # For each boolean column index b, pick 0/1 based on bool_probs[b].
                # If you want different logic (like "choose original or neighbor"?),
                # you can adapt here.
                n_bool_cols = len(self.bool_cols)
                synthetic_slice_excl = np.zeros((self.slice_size, n_bool_cols), dtype=np.float32)

                for col_idx_in_boolarray, bcol in enumerate(self.bool_cols):
                    p = bool_probs[bcol]  # Probability of 1 for that bool column
                    # Sample 0/1 for each time step in the slice
                    synthetic_slice_excl[:, col_idx_in_boolarray] = \
                        np.random.binomial(n=1, p=p, size=self.slice_size)
                
                # Combine numeric + boolean columns back in correct order
                # We have numeric_cols in synthetic_slice_incl
                # We have bool_cols in synthetic_slice_excl
                # We need to re-insert them into shape: (slice_size, total_channels)
                synthetic_slice = np.zeros((self.slice_size, len(all_cols)), dtype=np.float32)

                # Place numeric columns
                synthetic_slice[:, numeric_cols] = synthetic_slice_incl
                # Place boolean columns
                synthetic_slice[:, self.bool_cols] = synthetic_slice_excl

                synthetic_slices.append(synthetic_slice)

            # Concatenate slices into a full time series (2000, ch)
            full_series = np.concatenate(synthetic_slices, axis=0)
            syn_samples.append(full_series)

        return np.array(syn_samples)

    def fit_resample(self, X, y):
        """
        Perform TSMOTE oversampling.
        :param X: shape (N, 2000, ch)
        :param y: shape (N,)
        """
        y_int = y.astype(int)
        class_counts = np.bincount(y_int)
        minority_class = np.argmin(class_counts)
        majority_class = 1 - minority_class

        n_needed = class_counts[majority_class] - class_counts[minority_class]
        if n_needed <= 0:
            return X, y  # no oversampling needed

        # Suppose X has shape (N, 2000, ch). We'll assume 2000 is consistent with time_slices * slice_size.
        # We'll deduce slice_size
        self.slice_size = X.shape[1] // self.time_slices  # e.g. 2000/10=200

        # Get only minority samples
        minority_samples = X[y_int == minority_class]

        # ----- Compute distribution of booleans in the minority data ------
        # For each bool column b, compute fraction of 1s across the entire minority set
        bool_probs = {}
        if len(self.bool_cols) > 0:
            # shape is (N_minority, 2000, ch)
            # We'll flatten across time for each column to get overall fraction
            for bcol in self.bool_cols:
                col_values = minority_samples[:, :, bcol].flatten()
                p = col_values.mean()  # fraction of 1's
                bool_probs[bcol] = p
        # ------------------------------------------------------------------

        synthetic = self._generate_synthetic(minority_samples, bool_probs)

        # Ensure matching dimensions
        assert X.shape[1:] == synthetic.shape[1:], \
            f"Dimension mismatch: Original {X.shape[1:]}, Synthetic {synthetic.shape[1:]}"

        # Use only as many synthetic as needed
        synthetic = synthetic[:n_needed]

        # Concatenate
        X_resampled = np.concatenate([X, synthetic], axis=0)
        y_resampled = np.concatenate([y, [minority_class] * len(synthetic)], axis=0)
        return X_resampled, y_resampled
