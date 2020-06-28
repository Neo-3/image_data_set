import pandas as pd


BEST_FEATURES_INDEX_LIST = [1109, 1111, 1621, 1623, 2133, 4693, 4695, 5193, 5207, 5319, 8683, 8777, 9170, 9200, 9289, 9392, 9636, 9692, 10148, 10204, 10660, 10825, 11760, 12754, 12779, 13266, 13385, 13732, 13788, 14244, 14756, 14921, 15268, 16875, 16969, 17316, 17481,
                            17591, 17690, 17723, 17828, 17852, 17870, 17884, 18103, 18202, 18340, 18615, 18714, 18747, 18852, 19226, 19364, 21687, 21924, 21948, 22199, 22436, 22460, 22711, 22810, 22948, 22972, 23004, 23460, 23484, 23739, 25230, 25275, 25742, 26254, 26556, 27068]


def get_best_features(flatten_feature):
    best_features = [flatten_feature[i] for i in BEST_FEATURES_INDEX_LIST]
    return best_features


def _append_target(features_list, target):
    df = pd.DataFrame(features_list)
    target_list = []
    for i in range(len(df)):
        target_list.append(target)
    df["Target"] = target_list
    return df


def generate_csv(good_features, bad_features, output_path):
    df_good = _append_target(good_features, 1)
    df_bad = _append_target(bad_features, 0)
    df_to_csv = df_good.append(df_bad)
    df_to_csv.to_csv(output_path, encoding='utf-8', index=False)
