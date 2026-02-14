import argparse
import json

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True, help='City name (NYC, TKY, or CA)')
    args = parser.parse_args()

    city = args.city.upper()

    # Set data paths based on city
    if city == "CA":
        train_path = "dataset/Gowalla/gowalla_train.csv"
        output_path = "dataset/Gowalla/gowalla_user_checked_all.json"
    else:
        train_path = f"dataset/NYC and Tokyo Check-in/{city}/{city}_train.csv"
        output_path = f"dataset/NYC and Tokyo Check-in/{city}/{city}_user_checked_all.json"

    # Load training data
    train_df = pd.read_csv(train_path)

    # Build POI id to index mapping
    poi_ids = list(set(train_df['POI_id'].astype(str).tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Extract user check-in history
    user_checked = {}

    for traj_id in tqdm(set(train_df['trajectory_id'].tolist()), desc="Processing trajectories"):
        user_id = traj_id.split('_')[0]

        traj_df = train_df[train_df['trajectory_id'] == traj_id]
        poi_ids = traj_df['POI_id'].astype(str).to_list()
        poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]

        if user_id not in user_checked:
            user_checked[user_id] = list(set(poi_idxs))
        else:
            user_checked[user_id].extend(poi_idxs)
            user_checked[user_id] = list(set(user_checked[user_id]))

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(user_checked, f, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    main()
