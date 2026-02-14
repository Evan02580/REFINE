# REFINE: A Resource-Efficient LLM-based Approach for Next Top-K POI Recommendation

The repository introduces the implementation of **REFINE: A Resource-Efficient LLM-based Approach for Next Top-K POI Recommendation**. This work aims to solve the next Point-of-Interest (POI) recommendation problem with Large Language Models (LLMs). While LLMs are effective at capturing contextual information from historical check-in data, they face challenges such as high computational costs, input length constraints, and difficulty in producing ranked Top-K recommendations due to their generative nature. To overcome these issues, we propose **REFINE**, a method that uses embedding-based prompts, enabling LLMs to access necessary information efficiently without heavy computational demands. The proposed approach is evaluated on three real-world datasets and outperforms existing methods across all of them.

## Project Structure

```
REFINE
├── dataset/                      # Dataset folders
│   ├── GoWalla/
│   │   ├── CA_train.csv
│   │   └── CA_test.csv
│   └── NYC and Tokyo Check-in/
│       ├── NYC/                  # Contains the train and test dataset
│       │   ├── NYC_train.csv
│   	│   └── NYC_test.csv
│       └── TKY/
│           ├── TKY_train.csv
│   	    └── TKY_test.csv
├── poi2emb/                      # POI embedding files (generated)
│	├ NYC_poi2emb.json
│	├ TKY_poi2emb.json
│	└ CA_poi2emb.json
├── train.py                      # Main training script
├── utils.py                      # Evaluation metrics
├── generate_poi_embeddings.py    # Generate POI basic embeddings
└── generate_user_checkins.py     # Extract user check-in history from training data
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+ (HuggingFace)
- peft (for LoRA)
- pandas, numpy, scikit-learn, tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Implement

The dataset pre-processing follows the previous study ([CoMaPOI](https://github.com/Chips98/CoMaPOI)). NYC and TKY data are stored in the path `dataset/NYC and Tokyo Check-in/{city}`, CA data is located in `dataset/Gowalla`. Then process and train the datasets as following steps.

### 1. Generate POI Embeddings

Generate POI basic embeddings (take NYC as example):

```bash
python generate_poi_embeddings.py --city NYC
```

This generates `{city}_poi2emb.json` files under the `poi2emb` folder containing embeddings in format `[id, lat, lon, one_hot_category...]`.

### 2. Generate User Check-in Records

Extract user check-in history from training data:

```bash
python generate_user_checkins.py --city NYC
```

This generates `{city}_user_checked_all.json` files under the `dataset` folder containing the historical POIs visited by each user.

### 3. Model Training & Evaluation

Run the main training script:

```bash
# NYC dataset with Partial fine-tuning first 2 layers
python train.py --city NYC --batch 8 --device 0 --all_params 2
```

### Training Options

```
--batch: Batch size (default: 8)
--lr: Learning rate (default: 6e-5)
--device: GPU device ID (default: 0)
--all_params N: Partial fine-tune first N transformer layers
--lora: Enable LoRA fine-tuning
--layernorm N: Fine-tune first N layer norms (default: none)
--model_name: HuggingFace model name (default: meta-llama/Meta-Llama-3-8B)
--short_traj_thres: Minimum trajectory length (default: 3)
--nratio: Negative sampling ratio for BPR loss (default: 0.05)
--patience: LR scheduler patience (default: 5)
--weight_decay: Weight decay for optimizer (default: 0.01)
```

## Evaluation Metrics

The model evaluates on the following metrics during validation:
- Top-K Accuracy (@1, @5, @10, @20)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@5, @10, @20)

