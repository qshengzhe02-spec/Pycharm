

import os
import time
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH   = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH  = r"C:\Users\Qsz53\Desktop\测试集.xlsx"

OUT_DIR = r"F:\消融实验\ft"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

num_cols   = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]
label_col  = "岩性"
MAX_EPOCHS = 100
PATIENCE   = 12
TRC_EPOCHS = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FTTransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, attn_drop, ffn_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token*2), nn.GELU(), nn.Dropout(ffn_drop),
            nn.Linear(d_token*2, d_token), nn.Dropout(ffn_drop)
        )
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num, n_classes, d_token, n_blocks, n_heads, attn_drop, ffn_drop):
        super().__init__()
        self.token_emb = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.blocks = nn.Sequential(*[
            FTTransformerBlock(d_token, n_heads, attn_drop, ffn_drop) for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x_num, return_emb=False):
        B, N = x_num.shape
        tokens = self.token_emb(x_num.unsqueeze(-1))
        cls = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.blocks(x)
        emb = self.norm_out(x[:, 0, :])
        logits = self.head(emb)
        if return_emb:
            return logits, emb
        return logits

class TRCCorrector(nn.Module):
    def __init__(self, embed_dim, orth_weight=0.01):
        super().__init__()
        self.shift_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.coord_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.orth_weight = orth_weight

    def forward(self, z):
        z_re = z - self.shift_estimator(z)
        return self.coord_estimator(z_re)

    def orth_loss(self, emb):
        b, d = emb.shape
        gram = torch.matmul(emb, emb.T)
        I = torch.eye(b, device=emb.device)
        return self.orth_weight * torch.norm(gram - I, p="fro") / b

class FT_TRC_Model(nn.Module):
    def __init__(self, ft, trc, n_classes):
        super().__init__()
        self.ft = ft
        self.trc = trc
        self.head = nn.Linear(ft.token_emb.out_features, n_classes)
    def forward(self, x_num):
        _, emb = self.ft(x_num, return_emb=True)
        z_corr = self.trc(emb)
        return self.head(z_corr), z_corr


class LogDataset(Dataset):
    def __init__(self, X_num, y):
        self.X_num = torch.FloatTensor(X_num)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X_num)
    def __getitem__(self, idx):
        return self.X_num[idx], self.y[idx]


def load_data():
    df_train = pd.read_excel(TRAIN_PATH)
    df_val   = pd.read_excel(VAL_PATH)
    df_test  = pd.read_excel(TEST_PATH)
    X_train, y_train = df_train[num_cols].copy(), df_train[label_col].copy()
    X_val,   y_val   = df_val[num_cols].copy(),   df_val[label_col].copy()
    X_test,  y_test  = df_test[num_cols].copy(),  df_test[label_col].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print(f"✅训练集 {X_train.shape} | 验证集 {X_val.shape}")
    print(f"✅类别数 {n_classes}: {list(le.classes_)}")
    return X_train,y_train,X_val,y_val,X_test,y_test,n_classes,le

X_train,y_train,X_val,y_val,X_test,y_test,n_classes,le = load_data()
bs =32
train_loader = DataLoader(LogDataset(X_train,y_train), batch_size=bs, shuffle=True)
val_loader   = DataLoader(LogDataset(X_val,y_val),   batch_size=bs, shuffle=False)

unique, counts = np.unique(y_train, return_counts=True)
total = np.sum(counts)
soft_weight = torch.tensor(np.log1p(total/counts), dtype=torch.float32).to(device)


def evaluate_model(model, loader, is_trc=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xn, y in loader:
            xn = xn.to(device)
            if is_trc:
                logits, _ = model(xn)
            else:
                logits = model(xn)
            pred = logits.argmax(dim=1).cpu()
            y_pred.extend(pred.numpy())
            y_true.extend(y.cpu().numpy())
    acc = accuracy_score(y_true,y_pred)
    f1_macro = f1_score(y_true,y_pred,average="macro")
    return acc, f1_macro


def train_ft_only(params):
    start_t = time.perf_counter()
    ft = FTTransformer(len(num_cols), n_classes,
                       params["d_token"], params["n_blocks"],
                       params["n_heads"], params["attn_drop"], params["ffn_drop"]).to(device)
    opt = optim.AdamW(ft.parameters(), lr=params["lr"], weight_decay=params["wd"])
    loss_fn = nn.CrossEntropyLoss(weight=soft_weight)
    best_val_f1 = -1.0
    best_ft = ft
    patience = 0
    epoch_history = []
    for ep in range(MAX_EPOCHS):
        ft.train()
        train_loss = 0.0
        for xn, y in train_loader:
            xn, y = xn.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(ft(xn), y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        ft.eval()
        tr_pred, tr_true = [], []
        yv_pred, yv_true = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xn, y in train_loader:
                xn, y = xn.to(device), y.to(device)
                logits = ft(xn)
                tr_pred.extend(logits.argmax(1).cpu().numpy())
                tr_true.extend(y.cpu().numpy())
            for xn, y in val_loader:
                xn, y = xn.to(device), y.to(device)
                logits = ft(xn)
                val_loss += loss_fn(logits, y).item()
                yv_pred.extend(logits.argmax(1).cpu().numpy())
                yv_true.extend(y.cpu().numpy())
        train_acc = accuracy_score(tr_true, tr_pred)
        val_acc   = accuracy_score(yv_true, yv_pred)
        val_f1    = f1_score(yv_true, yv_pred, average="macro")
        epoch_history.append({
            "epoch": ep+1,
            "train_loss": round(train_loss / len(train_loader),4),
            "train_acc": round(train_acc,4),
            "val_loss": round(val_loss / len(val_loader),4),
            "val_acc": round(val_acc,4),
            "val_f1": round(val_f1,4),
        })
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_ft = ft
            patience =0
        else:
            patience +=1
            if patience >= PATIENCE:
                break
    cost = time.perf_counter() - start_t
    return best_ft, params, epoch_history, best_val_f1, cost


def train_trc_on_frozen_ft(ft_base, embed_dim):
    t0 = time.perf_counter()
    for p in ft_base.parameters():
        p.requires_grad = False
    trc = TRCCorrector(embed_dim=embed_dim).to(device)
    ft_trc = FT_TRC_Model(ft_base, trc, n_classes).to(device)
    opt_trc = optim.AdamW([p for p in ft_trc.parameters() if p.requires_grad], lr=1e-3)
    loss_cls = nn.CrossEntropyLoss(weight=soft_weight)
    for ep in range(TRC_EPOCHS):
        ft_trc.train()
        for xn, y in train_loader:
            xn, y = xn.to(device), y.to(device)
            logits, emb = ft_trc(xn)
            loss = loss_cls(logits, y) + trc.orth_loss(emb)
            opt_trc.zero_grad(); loss.backward(); opt_trc.step()
    trc_cost = time.perf_counter() - t0
    trc_train_acc, trc_train_f1 = evaluate_model(ft_trc, train_loader, is_trc=True)
    trc_val_acc, trc_val_f1     = evaluate_model(ft_trc, val_loader,   is_trc=True)
    return ft_trc, trc_cost, trc_train_acc, trc_train_f1, trc_val_acc, trc_val_f1


base_fix = {
    "d_token":32,
    "n_blocks":2,
    "n_heads":4,
    "lr":5e-4,
    "attn_drop":0.05,
    "ffn_drop":0.05,
    "wd":1e-4
}
full_candidate = {
    "d_token":    [16, 32, 64],
    "n_blocks":   [1, 2],
    "n_heads":    [2, 4],
    "lr":         [1e-3, 5e-4, 2e-4],
    "attn_drop":  [0.0, 0.05, 0.1],
    "ffn_drop":   [0.0, 0.05, 0.1],
    "wd":         [1e-5, 1e-4]
}

exp_setting_list = [
    {
        "exp_id":"Exp1",
        "exp_note":"Ablation_fix_dtoken=32",
        "fix_param":{"d_token": base_fix["d_token"]},
        "search_space":{k:full_candidate[k] for k in ["n_blocks","n_heads","lr","attn_drop","ffn_drop","wd"]}
    },
    {
        "exp_id":"Exp2",
        "exp_note":"Ablation_fix_nblocks=2",
        "fix_param":{"n_blocks": base_fix["n_blocks"]},
        "search_space":{k:full_candidate[k] for k in ["d_token","n_heads","lr","attn_drop","ffn_drop","wd"]}
    },
    {
        "exp_id":"Exp3",
        "exp_note":"Ablation_fix_nheads=4",
        "fix_param":{"n_heads": base_fix["n_heads"]},
        "search_space":{k:full_candidate[k] for k in ["d_token","n_blocks","lr","attn_drop","ffn_drop","wd"]}
    },
    {
        "exp_id":"Exp4",
        "exp_note":"Ablation_fix_lr=5e-4",
        "fix_param":{"lr": base_fix["lr"]},
        "search_space":{k:full_candidate[k] for k in ["d_token","n_blocks","n_heads","attn_drop","ffn_drop","wd"]}
    },
    {
        "exp_id":"Exp5",
        "exp_note":"Ablation_fix_attndrop=0.05",
        "fix_param":{"attn_drop": base_fix["attn_drop"]},
        "search_space":{k:full_candidate[k] for k in ["d_token","n_blocks","n_heads","lr","ffn_drop","wd"]}
    },
    {
        "exp_id":"Exp6",
        "exp_note":"Ablation_fix_ffndrop=0.05",
        "fix_param":{"ffn_drop": base_fix["ffn_drop"]},
        "search_space":{k:full_candidate[k] for k in ["d_token","n_blocks","n_heads","lr","attn_drop","wd"]}
    },
    {
        "exp_id":"Exp7",
        "exp_note":"Ablation_fix_wd=1e-4",
        "fix_param":{"wd": base_fix["wd"]},
        "search_space":{k:full_candidate[k] for k in ["d_token","n_blocks","n_heads","lr","attn_drop","ffn_drop"]}
    }
]


global_summary_list = []

for exp_cfg in exp_setting_list:
    exp_id = exp_cfg["exp_id"]
    exp_note = exp_cfg["exp_note"]
    fix_p = exp_cfg["fix_param"]
    search_space = exp_cfg["search_space"]
    print(f"\n>>>>>>>>>>>>>>开始 {exp_id}: {exp_note}；固定参数：{fix_p}")
    keys_sp = list(search_space.keys())
    values_sp = list(search_space.values())
    comb_list = list(itertools.product(*values_sp))
    param_grid = [ {**fix_p, **dict(zip(keys_sp, one_item))} for one_item in comb_list ]
    total_comb = len(param_grid)
    print(f"本组待运行参数组合总数：{total_comb}")

    all_tuning = []
    best_val_f1_exp = -1.0
    best_ft_model_exp = None
    best_param_exp = None
    search_start = time.perf_counter()

    for idx, p in enumerate(param_grid):
        print(f"-----组内参数 {idx+1}/{total_comb} ----- {p}")
        try:
            ft, param, hist, current_val_f1, param_cost = train_ft_only(p)
            tr_acc, tr_f1 = evaluate_model(ft, train_loader, is_trc=False)
            all_tuning.append({
                **p,
                "train_acc":round(tr_acc,4),
                "train_f1":round(tr_f1,4),
                "val_f1":round(current_val_f1,4),
                "best_epoch": len(hist),
                "param_train_time_s": round(param_cost,2),
                "epoch_history": hist
            })
            if current_val_f1>best_val_f1_exp:
                best_val_f1_exp = current_val_f1
                best_ft_model_exp = ft
                best_param_exp = p
        except Exception as e:
            print(f"❌本组参数训练异常跳过，error:{e}")

    grid_search_elapsed = time.perf_counter() - search_start

    embed_dim = best_param_exp["d_token"]
    ft_trc_model, trc_cost, trc_tr_acc, trc_tr_f1, trc_val_acc, trc_val_f1 = train_trc_on_frozen_ft(best_ft_model_exp, embed_dim)
    ft_tr_acc, ft_tr_f1 = evaluate_model(best_ft_model_exp, train_loader, is_trc=False)
    ft_val_acc, ft_val_f1 = evaluate_model(best_ft_model_exp, val_loader, is_trc=False)


    result_local_df = pd.DataFrame([
        {"model_type":"FT‑Transformer",
         "train_acc":ft_tr_acc,"train_f1":ft_tr_f1,
         "val_acc":ft_val_acc,"val_f1":ft_val_f1,
         "params":str(best_param_exp)},
        {"model_type":"FT‑Transformer+TRC",
         "train_acc":trc_tr_acc,"train_f1":trc_tr_f1,
         "val_acc":trc_val_acc,"val_f1":trc_val_f1,
         "params":str(best_param_exp)+"+TRC"}
    ])

 
    out_excel = os.path.join(OUT_DIR, f"{exp_id}_{exp_note}_寻优汇总.xlsx")
    sum_cols = list(full_candidate.keys()) + ["train_acc","train_f1","val_f1","best_epoch","param_train_time_s"]
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        summary_df = pd.DataFrame([{k:t.get(k,None) for k in sum_cols} for t in all_tuning])
        summary_df.to_excel(writer, sheet_name="参数汇总", index=False)
        for i, rec in enumerate(all_tuning):
            pd.DataFrame(rec["epoch_history"]).to_excel(writer, sheet_name=f"param_{i+1}_epoch", index=False)
        result_local_df.to_excel(writer, sheet_name="本组最优‑FT‑TRC结果", index=False)

    global_summary_list.append({
        "exp_id":exp_id,
        "exp_note":exp_note,
        "total_combinations":total_comb,
        "fix_param":str(fix_p),
        "best_FT_val_f1":round(ft_val_f1,4),
        "best_FT_TRC_val_f1":round(trc_val_f1,4),
        "grid_search_time_sec":round(grid_search_elapsed,2),
        "TRC_head_train_time_sec":round(trc_cost,2),
        "best_ft_params":str(best_param_exp)
    })
    print(f"⏱️ {exp_id}网格耗时 {grid_search_elapsed:.2f}s；TRC头训练 {trc_cost:.2f}s")
    print(f"✅FT val‑F1 {ft_val_f1:.4f} | FT+TRC val‑F1 {trc_val_f1:.4f}")
    print(f"✅保存文件 {out_excel}")


df_global = pd.DataFrame(global_summary_list)
csv_out = os.path.join(OUT_DIR, "Ablation_FT_TRC_Summary.csv")
df_global.to_csv(csv_out, index=False, encoding="utf‑8‑sig")
print("\n"*2,"="*95)
print("✅FT‑TRC完整消融7组全部结束！全局汇总表：")
print(df_global.to_string())
print(f"✅全局汇总csv路径：{csv_out}")
print("="*95)
