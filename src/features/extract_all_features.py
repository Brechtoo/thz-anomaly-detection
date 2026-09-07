from src.features.extract_baseline_features import (
    extract_baseline_features,
)

from src.features.extract_extension_block1_features import (
    extract_extension_block1_features,
)


def extract_all_features(file_path):

    baseline_features = extract_baseline_features(file_path)

    extension_features = extract_extension_block1_features(file_path)

    duplicate_features = (baseline_features.keys() & extension_features.keys())

    if duplicate_features:
        raise ValueError(f"doppelte feature-namen: {duplicate_features}")

    return {**baseline_features, **extension_features}