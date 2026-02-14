import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import top_k_acc_last_timestep, mAP_metric_last_timestep, MRR_metric_last_timestep
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """Time2Vec encoding function."""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


def t2norm_batch(time_strs):
    """Convert time strings to normalized time of day (48 time slots)."""
    norm_in_day = []
    for time_str in time_strs:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        norm_in_day.append((dt.hour * 2 + dt.minute // 30) / 48)
    return norm_in_day


def t2dow_batch(time_strs):
    """Convert time strings to normalized day of week."""
    day_of_week = []
    for time_str in time_strs:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        day_of_week.append(dt.weekday() / 7)
    return day_of_week


def real_t2v_batch(time_strs):
    """Convert time strings to integer components."""
    time_ints = []
    for time_str in time_strs:
        date_str, time_str = time_str.split()
        ymd = list(map(int, date_str.split('-')))
        hms = list(map(int, time_str.split(':')))
        time_ints.append(ymd + hms)
    return time_ints


def real_v2t_batch(time_ints):
    """Convert integer components back to time strings."""
    time_strs = []
    for time_int in time_ints:
        time_strs.append(
            f"{time_int[0]:04d}-{time_int[1]:02d}-{time_int[2]:02d} {time_int[3]:02d}:{time_int[4]:02d}:{time_int[5]:02d}")
    return time_strs


def calculate_dcg(rel_list, k):
    """Calculate DCG at k."""
    dcg = 0
    for i in range(min(k, len(rel_list))):
        rel = rel_list[i]
        dcg += rel / math.log2(i + 2)
    return dcg


def calculate_ndcg(rel_list, k):
    """Calculate NDCG at k."""
    dcg = calculate_dcg(rel_list, k)
    ideal_rel_list = sorted(rel_list, reverse=True)
    idcg = calculate_dcg(ideal_rel_list, k)
    return dcg / idcg if idcg > 0 else 0


class SineActivation(nn.Module):
    """Sine activation for Time2Vec encoding."""
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class NextPOILLM(nn.Module):
    """Next POI recommendation model based on LLM."""
    def __init__(self, args, poi_id2idx_dict, poi_idx2id_dict, cats_num=200, user_num=100, num_poi=4980):
        super(NextPOILLM, self).__init__()

        self.negative_num = int(num_poi * args.nratio)
        self.poi_id2idx_dict = poi_id2idx_dict
        self.poi_idx2id_dict = poi_idx2id_dict

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none"
        )

        self.poi_num = len(poi_idx2id_dict)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if "llama" in args.model_name:
            self.llama = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        else:
            self.llama = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        self.llm_dim = self.llama.lm_head.weight.size(1)

        # Time encoding
        self.time_size = 1
        self.time_dim = self.llm_dim // 2
        self.time = SineActivation(self.time_size, self.time_dim).bfloat16()
        self.day_of_week = SineActivation(self.time_size, self.time_dim).bfloat16()

        if args.lora:
            self.llama = get_peft_model(self.llama, self.lora_config)
            self.llama.base_model.model.lm_head = nn.Linear(self.llm_dim, self.poi_num, bias=False)
        else:
            self.llama.lm_head = nn.Sequential(
                nn.Linear(self.llm_dim, self.llm_dim // 2, bias=False),
                nn.Linear(self.llm_dim // 2, self.poi_num, bias=False)
            )

        self.llama.config.vocab_size = self.poi_num
        self.softmax = nn.Softmax(dim=-1)

        # Load POI embeddings
        with open(f"./poi2emb/{args.city}_poi2emb.json", 'r', encoding='utf-8') as f:
            poi2emb = json.load(f)
        poi2emb = {p: poi2emb[p][1:] for p in poi_idx2id_dict.values()}
        lat_lon = torch.tensor([poi[:2] for poi in poi2emb.values()], dtype=torch.float64)

        with open(args.checked_record, 'r', encoding='utf-8') as f:
            self.user_checked = json.load(f)

        # Normalize lat/lon
        lat_min, lat_max = lat_lon[:, 0].min(), lat_lon[:, 0].max()
        lon_min, lon_max = lat_lon[:, 1].min(), lat_lon[:, 1].max()
        pos_emb = torch.zeros_like(lat_lon)
        pos_emb[:, 0] = (lat_lon[:, 0] - lat_min) / (lat_max - lat_min)
        pos_emb[:, 1] = (lat_lon[:, 1] - lon_min) / (lon_max - lon_min)

        poi2emb_tensor = torch.tensor([p[2:] for p in poi2emb.values()])
        self.poi2emb = torch.cat((pos_emb, poi2emb_tensor), dim=-1).to(args.device).bfloat16()

        self.user2emb = nn.Embedding(user_num, self.llm_dim)
        self.poi_dim = self.poi2emb.size(1)
        self.transfer = nn.Linear(self.poi_dim, self.llm_dim)
        self.all_poi = self.poi2emb.clone()

    def embpreprocess(self, user, pois, times, device):
        """Process embeddings for input sequence."""
        users = torch.tensor(int(user)).to(device).repeat(len(pois))
        times = real_v2t_batch(times)

        users_emb = self.user2emb(users)
        pois_emb = self.transfer(torch.stack([self.poi2emb[p] for p in pois], dim=0).to(device).bfloat16())
        dow_emb = self.day_of_week(torch.tensor(t2dow_batch(times)).to(device).unsqueeze(1).bfloat16())
        nid_emb = self.time(torch.tensor(t2norm_batch(times)).to(device).unsqueeze(1).bfloat16())
        time_emb = torch.cat((dow_emb, nid_emb), dim=-1)
        pad = torch.zeros_like(pois_emb).to(device).bfloat16()

        all_poi_emb = self.transfer(self.all_poi)

        # Previous POI
        pre_poi_emb = pois_emb[:-1, :]
        pre_poi_emb = torch.cat([all_poi_emb.mean(0).unsqueeze(0), pre_poi_emb], dim=0)

        mask_label = torch.ones(len(pois), 1) * -100
        current_emb = torch.stack((users_emb, pre_poi_emb, time_emb, pois_emb, pad), dim=1).view(-1, self.llm_dim)[:-1, :]
        label_cur = torch.stack([mask_label, mask_label, mask_label, torch.tensor(pois).unsqueeze(1), mask_label], dim=1).view(-1)[:-1]

        label_cur[:5] = -100

        input_emb = current_emb
        label = label_cur

        return input_emb, label, torch.ones_like(label)

    def forward(self, input, label, mask, users, device):
        """Forward pass for training."""
        hidden = self.llama(inputs_embeds=input.bfloat16(), attention_mask=mask.to(device),
                            labels=label.long().to(device), output_hidden_states=True)
        last_pos = [(sample == 1).nonzero(as_tuple=False)[-1].item() - 1 for sample in mask]

        all_pos2 = [((sample != -100).nonzero(as_tuple=False).squeeze(1)).tolist() for sample in label]
        user = [users[j] for j, a in enumerate(all_pos2) for i in range(len(a))]

        hls = hidden.loss

        logits = hidden.logits.view(-1, self.poi_num)
        labels = label.view(-1)
        all_pos = ((labels != -100).nonzero(as_tuple=False).squeeze(1) - 1).tolist()
        all_pos2 = ((labels != -100).nonzero(as_tuple=False).squeeze(1)).tolist()
        logits = logits[all_pos, :]
        labels = labels[all_pos2].long()
        neg_indices = torch.zeros((logits.size(0), self.negative_num), dtype=torch.long, device=device)

        # BPR negative sampling
        for i in range(logits.size(0)):
            _label = labels[i].item()
            _logits = logits[i, :].clone()
            out_pois = torch.tensor([np for np in self.user_checked[user[i]]]).to(device).long()
            _logits[out_pois] = 0
            _, large_index = torch.topk(_logits, dim=0, k=int(self.negative_num), largest=True)
            neg_indices[i] = large_index

        pos_logits = logits.gather(1, labels.unsqueeze(1).to(device))
        neg_logits = logits.gather(1, neg_indices)
        pos_logits = pos_logits.expand_as(neg_logits)

        margin_loss = torch.log(torch.sigmoid(pos_logits - neg_logits))
        rank_loss = -margin_loss.mean()

        return hls + rank_loss, hidden.logits[[i for i in range(input.size(0))], last_pos, :]

    def testemb(self, input, mask, users, label, device):
        """Generate predictions for testing."""
        out = []
        last_pos = [(sample == 1).nonzero(as_tuple=False)[-1].item() for sample in mask]

        for i in range(input.size(0)):
            _input = input[i, :last_pos[i]].unsqueeze(0)
            _mask = mask[i, :last_pos[i]].unsqueeze(0)

            _out = self.llama.generate(inputs_embeds=_input.bfloat16(), attention_mask=_mask.to(device),
                                       max_new_tokens=1, return_dict_in_generate=True, output_scores=True,
                                       do_sample=False)

            _out = _out.scores[0]
            out.append(_out)
        out = torch.cat(out, 0)
        return out


class TrajectoryDatasetTrain(Dataset):
    """Training dataset for trajectories."""
    def __init__(self, train_df, poi_id2idx_dict, args):
        self.df = train_df
        self.poi_id2idx_dict = poi_id2idx_dict
        self.args = args
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
            user_id = traj_id.split('_')[0]

            traj_df = train_df[train_df['trajectory_id'] == traj_id]
            poi_ids = traj_df['POI_id'].astype(str).to_list()
            poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
            time_feature = traj_df[args.time_feature].to_list()

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i][:-6]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1][:-6]))

            if len(input_seq) < args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self):
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])


class TrajectoryDatasetVal(Dataset):
    """Validation/test dataset for trajectories."""
    def __init__(self, df, poi_id2idx_dict, args):
        self.df = df
        self.poi_id2idx_dict = poi_id2idx_dict
        self.args = args
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for user_id in tqdm(set(df['user_id'].tolist())):
            traj_df = df[df['user_id'] == user_id]
            poi_ids = traj_df['POI_id'].astype(str).to_list()
            poi_idxs = []
            time_feature = traj_df[args.time_feature].to_list()
            time_feature_real = []

            for i, each in enumerate(poi_ids):
                if each in poi_id2idx_dict.keys():
                    poi_idxs.append(poi_id2idx_dict[each])
                    time_feature_real.append(time_feature[i])
                else:
                    continue

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature_real[i][:-6]))
                label_seq.append((poi_idxs[i + 1], time_feature_real[i + 1][:-6]))

            if len(input_seq) < args.short_traj_thres:
                continue

            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)
            self.traj_seqs.append(user_id)

    def __len__(self):
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-5, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='LR scheduler patience')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='LR scheduler factor')
    parser.add_argument('--lora', action='store_true', help='Enable LoRA')
    parser.add_argument('--all_params', type=int, default=0, help='Fine-tune first N layers')
    parser.add_argument('--layernorm', type=int, default=32, help='Fine-tune first N layernorms')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B", help='LLM model name')
    parser.add_argument('--short_traj_thres', type=int, default=3, help='Short trajectory threshold')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--city', type=str, default="NYC", help='City name')
    parser.add_argument('--nratio', type=float, default=0.05, help='Negative sampling ratio')
    args = parser.parse_args()

    args.epochs = 200
    args.city = args.city.upper()

    # Set data paths based on city
    if args.city == "NYC" or args.city == "TKY":
        args.data_train = f'dataset/NYC and Tokyo Check-in/{args.city}/{args.city}_train.csv'
        args.data_test = f'dataset/NYC and Tokyo Check-in/{args.city}/{args.city}_test.csv'
        args.checked_record = f'dataset/NYC and Tokyo Check-in/{args.city}/{args.city}_user_checked_all.json'
    elif args.city == "CA":
        args.data_train = 'dataset/Gowalla/gowalla_train.csv'
        args.data_test = 'dataset/Gowalla/gowalla_test.csv'
        args.checked_record = 'dataset/Gowalla/gowalla_user_checked_all.json'

    args.time_feature = 'local_time'
    if args.city == "CA":
        args.time_feature = 'checkin_time'

    args.data_val = args.data_test
    device = torch.device(f'cuda:{args.device}')
    args.device = device

    # Load data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build mappings
    poi_ids = list(set(train_df['POI_id'].astype(str).tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
    poi_idx2id_dict = {v: k for k, v in poi_id2idx_dict.items()}
    print(f"Unique POI Number: {len(poi_ids)}")

    cat_ids = list(set(train_df['POI_catname'].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))
    print(f"Unique Cat Number: {len(cat_ids)}")

    poi_idx2cat_idx_dict = {}
    nodes_df = train_df[['POI_id', 'POI_catname']].drop_duplicates().reset_index(drop=True)
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[str(row['POI_id'])]] = cat_id2idx_dict[row['POI_catname']]

    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
    print(f"Unique User Number: {len(user_ids)}")

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Start training')

    # Create datasets
    train_dataset = TrajectoryDatasetTrain(train_df, poi_id2idx_dict, args)
    val_dataset = TrajectoryDatasetVal(val_df, poi_id2idx_dict, args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False,
                              pin_memory=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False,
                            pin_memory=True, collate_fn=lambda x: x)

    user_num = max([int(u) for u in list(user_id2idx_dict.keys())]) + 1
    cat_num = len(cat_ids)
    model = NextPOILLM(args, poi_id2idx_dict, poi_idx2id_dict, cats_num=cat_num, user_num=user_num).to(device).bfloat16()
    print(model)

    # Configure trainable parameters
    if args.lora:
        for name, param in model.named_parameters():
            if "llama" in name:
                if "lora" in name or "lm_head" in name or "embed_tokens" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
    elif args.all_params != 0:
        for name, param in model.named_parameters():
            if "llama" in name:
                if "layers" in name:
                    param.requires_grad = True
                    if int(name.split('.')[3]) >= args.all_params:
                        param.requires_grad = False
                elif "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if "llama" in name:
                if "input_layernorm" in name:
                    param.requires_grad = True
                    if int(name.split('.')[3]) >= args.layernorm:
                        param.requires_grad = False
                elif "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience,
                                                              factor=args.lr_scheduler_factor)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    best_score = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"{'*' * 50} Epoch:{epoch:03d} {'*' * 50}")
        model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []

        for batch in tqdm(train_loader, total=len(train_loader)):
            batch_input_list = []
            batch_label_list = []
            batch_mask_list = []
            batch_user = []
            batch_seq_labels_poi = []

            for sample in batch:
                traj_id = sample[0]
                user_id = traj_id.split('_')[0]
                batch_user.append(user_id)

                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))

                input_seq_time = real_t2v_batch([each[1] for each in sample[1]])
                label_seq_time = real_t2v_batch([each[1] for each in sample[2]])

                input, train_label, mask = model.embpreprocess(user_id, input_seq + [label_seq[-1]],
                                                               input_seq_time + [label_seq_time[-1]], device)
                batch_input_list.append(input)
                batch_label_list.append(train_label)
                batch_mask_list.append(mask)

            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-100)
            y_poi = label_padded_poi.to(device=device, dtype=torch.long)

            batch_input_pad = pad_sequence(batch_input_list, batch_first=True, padding_value=2)
            batch_label_pad = pad_sequence(batch_label_list, batch_first=True, padding_value=-100)
            batch_mask_pad = pad_sequence(batch_mask_list, batch_first=True, padding_value=0)

            optimizer.zero_grad()
            loss, y_pred_poi = model(batch_input_pad, batch_label_pad, batch_mask_pad, batch_user, device)

            loss.backward()
            optimizer.step()

            # Calculate metrics
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()

            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0

            for label_pois, pred_pois in zip(batch_label_pois, batch_pred_pois):
                idx = -1
                clabel = label_pois[idx]
                if clabel == -100:
                    while clabel == -100:
                        idx -= 1
                        clabel = label_pois[idx]

                label_pois = [clabel]
                pred_pois = np.expand_dims(pred_pois, 0)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())

        # Validation
        model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_ndcg5_list = []
        val_batches_ndcg10_list = []
        val_batches_ndcg20_list = []

        for batch in tqdm(val_loader, total=len(val_loader)):
            batch_input_list = []
            batch_label_list = []
            batch_mask_list = []
            batch_user = []
            batch_seq_labels_poi = []
            batch_seq_lens = []

            for sample in batch:
                traj_id = sample[0]
                user_id = traj_id
                batch_user.append(user_id)

                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))

                input_seq_time = real_t2v_batch([each[1] for each in sample[1]])
                label_seq_time = real_t2v_batch([each[1] for each in sample[2]])

                input, train_label, mask = model.embpreprocess(user_id, input_seq + [label_seq[-1]],
                                                               input_seq_time + [label_seq_time[-1]], device)
                batch_input_list.append(input)
                batch_label_list.append(train_label)
                batch_mask_list.append(mask)
                batch_seq_lens.append(len(input_seq))

            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-100)
            y_poi = label_padded_poi.to(device=device, dtype=torch.long)

            batch_input_pad = pad_sequence(batch_input_list, batch_first=True, padding_value=2)
            batch_mask_pad = pad_sequence(batch_mask_list, batch_first=True, padding_value=0)

            y_pred_poi = model.testemb(batch_input_pad, batch_mask_pad, batch_user, label_padded_poi, device)

            # Get last label for each sequence
            last_y_poi = []
            for i in range(y_poi.size(0)):
                idx = -1
                clabel = y_poi[i][idx]
                if clabel == -100:
                    while clabel == -100:
                        idx -= 1
                        clabel = y_poi[i][idx]
                last_y_poi.append([clabel])
            last_y_poi = torch.tensor(last_y_poi).to(device)

            loss_poi = criterion_poi(y_pred_poi, last_y_poi.squeeze(1))
            loss = loss_poi

            # Calculate metrics
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()

            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            ndcg_sum_5 = 0
            ndcg_sum_10 = 0
            ndcg_sum_20 = 0
            mAP20 = 0
            mrr = 0

            for label_pois, pred_pois in zip(batch_label_pois, batch_pred_pois):
                idx = -1
                clabel = label_pois[idx]
                if clabel == -100:
                    while clabel == -100:
                        idx -= 1
                        clabel = label_pois[idx]

                label_pois = [clabel]
                pred_pois = np.expand_dims(pred_pois, 0)

                pred_ids = pred_pois[0].argsort()[-20:][::-1]
                rel_list_5 = [1 if pid == label_pois[0] else 0 for pid in pred_ids[:5]]
                rel_list_10 = [1 if pid == label_pois[0] else 0 for pid in pred_ids[:10]]
                rel_list_20 = [1 if pid == label_pois[0] else 0 for pid in pred_ids[:20]]

                ndcg_5 = calculate_ndcg(rel_list_5, 5)
                ndcg_10 = calculate_ndcg(rel_list_10, 10)
                ndcg_20 = calculate_ndcg(rel_list_20, 20)

                ndcg_sum_5 += ndcg_5
                ndcg_sum_10 += ndcg_10
                ndcg_sum_20 += ndcg_20

                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_ndcg5_list.append(ndcg_sum_5 / len(batch_label_pois))
            val_batches_ndcg10_list.append(ndcg_sum_10 / len(batch_label_pois))
            val_batches_ndcg20_list.append(ndcg_sum_20 / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)

        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_ndcg5 = np.mean(val_batches_ndcg5_list)
        epoch_val_ndcg10 = np.mean(val_batches_ndcg10_list)
        epoch_val_ndcg20 = np.mean(val_batches_ndcg20_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)

        lr_scheduler.step(epoch_val_loss)

        print(f"[Epoch {epoch}/{args.epochs}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
              f"train_loss:{epoch_train_loss:.4f}, "
              f"train_mAP20:{epoch_train_mAP20:.4f}, "
              f"train_mrr:{epoch_train_mrr:.4f}\n"
              f"val_loss: {epoch_val_loss:.4f}, "
              f"val_mAP20:{epoch_val_mAP20:.4f}, "
              f"val_mrr:{epoch_val_mrr:.4f}\n"
              f"|{epoch:3}/{args.epochs}|  Top1  |  Top5  | Top10  | Top20  |\n"
              f"|-------|--------|--------|--------|--------|\n"
              f"| train | {epoch_train_top1_acc:.4f} | {epoch_train_top5_acc:.4f} | {epoch_train_top10_acc:.4f} | {epoch_train_top20_acc:.4f} |\n"
              f"|  val  | {epoch_val_top1_acc:.4f} | {epoch_val_top5_acc:.4f} | {epoch_val_top10_acc:.4f} | {epoch_val_top20_acc:.4f} |\n"
              f"|-NDCG--| 5:{epoch_val_ndcg5:.4f} | 10:{epoch_val_ndcg10:.4f} | 20:{epoch_val_ndcg20:.4f}  |\n")

        overall = epoch_val_top5_acc + epoch_val_top10_acc + epoch_val_ndcg5 + epoch_val_ndcg10
        if overall > best_score:
            best_score = overall
            best_epoch = epoch
            print("====================================== BEST Epoch! ======================================")
        else:
            print(f"-------------------------------------- Best at {best_epoch} --------------------------------------")


if __name__ == '__main__':
    main()
