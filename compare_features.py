# File: compare_features.py
import pickle
import numpy as np
from scipy.spatial.distance import cdist # Efficient pairwise distance calculation
import argparse

def load_and_flatten_features(filename: str) -> np.ndarray:
    """Loads pickled features and returns them as a single NumPy array."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from {filename}. Found {len(data)} track IDs.")
    except FileNotFoundError:
        print(f"Error: File not found '{filename}'")
        return None
    except Exception as e:
        print(f"Error loading pickle file '{filename}': {e}")
        return None

    all_features_list = []
    for track_id, feature_list in data.items():
        # feature_list contains multiple embeddings for the track_id
        if isinstance(feature_list, list) and len(feature_list) > 0:
             # Assuming each element in feature_list is a numpy array (embedding)
            all_features_list.extend(feature_list)
        else:
            print(f"Warning: Track ID {track_id} has unexpected data format or is empty. Skipping.")


    if not all_features_list:
        print(f"Error: No valid features extracted from '{filename}'.")
        return None

    # Convert list of numpy arrays to a single 2D numpy array
    try:
        feature_array = np.vstack(all_features_list).astype(np.float32)
        print(f"Extracted {feature_array.shape[0]} features with dimension {feature_array.shape[1]} from {filename}.")
        return feature_array
    except ValueError as e:
         print(f"Error stacking features from '{filename}'. Check if all embeddings have the same dimension: {e}")
         # Attempt to filter based on common dimension if possible (advanced)
         # For now, just return None
         return None
    except Exception as e:
        print(f"Unexpected error processing features from '{filename}': {e}")
        return None


def calculate_and_print_stats(distances: np.ndarray, metric_name: str):
    """Calculates and prints summary statistics for a distance matrix."""
    if distances is None or distances.size == 0:
        print(f"Cannot calculate stats for {metric_name}, distances array is empty.")
        return

    flat_distances = distances.flatten()

    print(f"\n--- {metric_name} Distance Statistics ---")
    print(f"  Shape of distance matrix: {distances.shape}")
    print(f"  Number of comparisons: {flat_distances.size}")
    if flat_distances.size > 0:
        print(f"  Minimum: {np.min(flat_distances):.4f}")
        print(f"  Maximum: {np.max(flat_distances):.4f}")
        print(f"  Mean:    {np.mean(flat_distances):.4f}")
        print(f"  Median:  {np.median(flat_distances):.4f}")
        print(f"  Std Dev: {np.std(flat_distances):.4f}")
    else:
         print("  No distances to analyze.")

def main():
    parser = argparse.ArgumentParser(description="Compare feature embeddings from two saved pickle files.")
    parser.add_argument("file1", help="Path to the first feature file (e.g., features_run1.pkl).")
    parser.add_argument("file2", help="Path to the second feature file (e.g., features_run2.pkl).")
    args = parser.parse_args()

    print("Loading features...")
    features1 = load_and_flatten_features(args.file1)
    features2 = load_and_flatten_features(args.file2)

    if features1 is None or features2 is None:
        print("Could not load features from one or both files. Exiting.")
        return

    if features1.shape[0] == 0 or features2.shape[0] == 0:
        print("One or both feature sets are empty after processing. Exiting.")
        return

    if features1.shape[1] != features2.shape[1]:
        print(f"Error: Feature dimensions do not match! "
              f"File 1: {features1.shape[1]}, File 2: {features2.shape[1]}. Cannot compute distances.")
        return

    print("\nCalculating pairwise distances...")

    # Calculate Euclidean Distances
    print("Calculating Euclidean distances...")
    try:
        euclidean_distances = cdist(features1, features2, metric='euclidean')
        calculate_and_print_stats(euclidean_distances, "Euclidean")
    except Exception as e:
        print(f"Error calculating Euclidean distances: {e}")


    # Calculate Cosine Distances (1 - Cosine Similarity)
    print("\nCalculating Cosine distances...")
    try:
        # Note: cdist 'cosine' calculates 1 - similarity. Lower is better match.
        cosine_distances = cdist(features1, features2, metric='cosine')
        calculate_and_print_stats(cosine_distances, "Cosine")
    except Exception as e:
        print(f"Error calculating Cosine distances: {e}")

    print("\nComparison finished.")

if __name__ == "__main__":
    main()