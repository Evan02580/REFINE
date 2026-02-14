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
        output_path = "poi2emb/CA_poi2emb.json"
    else:
        train_path = f"dataset/NYC and Tokyo Check-in/{city}/{city}_train.csv"
        output_path = f"poi2emb/{city}_poi2emb.json"

    # Load training data
    train_df = pd.read_csv(train_path)

    # Create category to index mapping
    poi_ids = list(set(train_df['POI_id'].astype(str).tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
    print(f"Unique POI Number: {len(poi_ids)}")

    # Cat id to index
    cat_ids = list(set(train_df['POI_catname'].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))
    print(f"Unique Cat Number: {len(cat_ids)}")

    poi_idx2cat_idx_dict = {}
    nodes_df = train_df[['POI_id', 'POI_catname']].drop_duplicates().reset_index(drop=True)

    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[str(row['POI_id'])]] = cat_id2idx_dict[row['POI_catname']]


    # Create poi_dict
    cat2emb = [
        [1 if k == j else 0 for j in range(len(cat_ids))]
        for k in range(len(cat_ids))
    ]
    print("Reading POI data...")
    i = 0
    poi2emb = {}

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        poi_cat = cat_id2idx_dict[row["POI_catname"]]
        poi_id = row["POI_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        if poi_id not in poi2emb:
            i += 1
            poi2emb[poi_id] = [i, lat, lon] + cat2emb[poi_cat]
        else:
            poi_idx = poi2emb[poi_id][0]
            poi2emb[poi_id] = [poi_idx, lat, lon] + cat2emb[poi_cat]

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(poi2emb, f)


if __name__ == '__main__':
    main()
