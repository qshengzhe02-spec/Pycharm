# -*- coding: utf-8 -*-
""" FT-Transformer + TRC（表征矫正模块）岩性多分类
✅ 手写FT模型（返回embedding），不依赖tab-transformer-pytorch接口
✅ 【更新】真正笛卡尔积网格搜索：定义各参数候选列表，自动生成全部参数组合
✅ 步骤：网格搜索选最优FT → 冻结FT → 训练TRC矫正头
✅ 输出：原生FT 与 FT+TRC 两组对比指标，保存Excel
✅ 每组参数遍历完成后自动保存调参汇总+每个epoch训练过程Excel
✅ 计时模块：每组耗时、网格搜索总耗时、TRC耗时、评估耗时
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import itertools # 新增：笛卡尔积生成参数组合
import warnings
warnings.filterwarnings("ignore")

# ====================== 【配置区，自行修改】 ======================
TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH   = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH  = r"C:\Users\Qsz53\Desktop\测试集.xlsx"
num_cols   = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]
label_col  = "岩性"
MAX_EPOCHS = 100
TRC_EPOCHS = 60
PATIENCE   = 12
MAX_COMBINATION = 20  # 限制最多跑多少组，防止组合爆炸，调大/去掉都可以
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 手写FT-Transformer ======================
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

# ====================== TRC 矫正模块 ======================
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

# ====================== 数据集 ======================
class LogDataset(Dataset):
    def __init__(self, X_num, y):
        self.X_num = torch.FloatTensor(X_num)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X_num)
    def __getitem__(self, idx):
        return self.X_num[idx], self.y[idx]

# ====================== 数据加载 ======================
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

    print(f"✅训练集 {X_train.shape} | 验证集 {X_val.shape} | 测试集 {X_test.shape}")
    print(f"✅类别数 {n_classes}: {list(le.classes_)}")
    return X_train,y_train,X_val,y_val,X_test,y_test,n_classes,le

X_train,y_train,X_val,y_val,X_test,y_test,n_classes,le = load_data()
bs = 32
train_loader = DataLoader(LogDataset(X_train,y_train), batch_size=bs, shuffle=True)
val_loader   = DataLoader(LogDataset(X_val,y_val),   batch_size=bs, shuffle=False)
test_loader  = DataLoader(LogDataset(X_test,y_test), batch_size=bs, shuffle=False)

# ====================== Soft类别权重 ======================
unique, counts = np.unique(y_train, return_counts=True)
total = np.sum(counts)
soft_weight = torch.tensor(np.log1p(total/counts), dtype=torch.float32).to(device)

# ====================== 【真正网格搜索：定义各超参候选列表】 ======================
search_space = {
    "d_token":    [16, 32, 64],
    "n_blocks":   [1, 2],
    "n_heads":    [2, 4],
    "lr":         [1e-3, 5e-4, 2e-4],
    "attn_drop":  [0.0, 0.05, 0.1],
    "ffn_drop":   [0.0, 0.05, 0.1],
    "wd":         [1e-5, 1e-4]
}
# 生成笛卡尔积全部组合
keys = list(search_space.keys())
values = list(search_space.values())
all_combs = list(itertools.product(*values))
param_grid = [dict(zip(keys, item)) for item in all_combs]

# 限制最大组合数量，防止跑爆机器
# if len(param_grid) > MAX_COMBINATION:
#     param_grid = param_grid[:MAX_COMBINATION]
print(f"✅ 网格搜索生成总组合数：{len(all_combs)}，本次实际运行：{len(param_grid)}")
print("="*50)

# ====================== 评估函数 ======================
def evaluate(model, loader, is_trc=False):
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
            y_pred.extend(pred.numpy()); y_true.extend(y.cpu().numpy())
    return accuracy_score(y_true,y_pred), f1_score(y_true,y_pred,average="macro")

# ====================== 训练FT函数 ======================
def train_ft_only(params):
    start_t = time.perf_counter()
    ft = FTTransformer(len(num_cols), n_classes, params["d_token"], params["n_blocks"],
                       params["n_heads"], params["attn_drop"], params["ffn_drop"]).to(device)
    opt = optim.AdamW(ft.parameters(), lr=params["lr"], weight_decay=params["wd"])
    loss_fn = nn.CrossEntropyLoss(weight=soft_weight)
    best_val_f1 = -1
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
            loss.backward(); opt.step()
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
            "epoch":     ep + 1,
            "train_loss": round(train_loss / len(train_loader), 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss / len(val_loader), 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; best_ft = ft; patience = 0
        else:
            patience += 1
            if patience >= PATIENCE: break
    cost = time.perf_counter() - start_t
    return best_ft, params, epoch_history, cost

# ====================== 网格搜索主循环 ======================
grid_search_start = time.perf_counter()
best_ft_model, best_params = None, None
best_global_val_f1 = -1
all_tuning = []

for idx, p in enumerate(param_grid):
    print(f"\n========== 正在训练第 {idx+1}/{len(param_grid)} 组参数 ==========")
    ft, param, hist, param_cost = train_ft_only(p)
    tr = evaluate(ft, train_loader)
    va = evaluate(ft, val_loader)
    current_val_f1 = va[1]

    all_tuning.append({
        **p,
        "train_acc": round(tr[0], 4), "train_f1": round(tr[1], 4),
        "val_acc":   round(va[0], 4), "val_f1":   round(va[1], 4),
        "train_val_acc_diff": round(tr[0] - va[0], 4),
        "best_epoch": len(hist),
        "param_train_time_s": round(param_cost,2),
        "epoch_history": hist,
    })
    print(f"参数{p} 完成 | val_f1={current_val_f1:.4f} | 本组耗时{param_cost:.2f}s")

    if current_val_f1 > best_global_val_f1:
        best_global_val_f1 = current_val_f1
        best_ft_model = ft
        best_params = p

grid_search_total_time = time.perf_counter() - grid_search_start
print(f"\n✅ 网格搜索全部结束！总耗时：{grid_search_total_time:.2f} s ({grid_search_total_time/60:.2f} min)")
print("\n✅ 最优FT参数:", best_params)
print(f"✅ 最优验证集val_f1:{best_global_val_f1:.4f}")

# ====================== 冻结FT，训练TRC矫正模块 ======================
trc_start = time.perf_counter()
for param in best_ft_model.parameters():
    param.requires_grad = False

embed_dim = best_params["d_token"]
trc = TRCCorrector(embed_dim=embed_dim).to(device)
ft_trc_model = FT_TRC_Model(best_ft_model, trc, n_classes).to(device)
opt_trc = optim.AdamW([p for p in ft_trc_model.parameters() if p.requires_grad], lr=1e-3)
loss_cls = nn.CrossEntropyLoss(weight=soft_weight)

for ep in range(TRC_EPOCHS):
    ft_trc_model.train()
    for xn, y in train_loader:
        xn, y = xn.to(device), y.to(device)
        logits, emb = ft_trc_model(xn)
        loss = loss_cls(logits, y) + trc.orth_loss(emb)
        opt_trc.zero_grad(); loss.backward(); opt_trc.step()
trc_total_time = time.perf_counter() - trc_start
print(f"\n✅ TRC模块训练完成，耗时：{trc_total_time:.2f} s")

# ====================== 评估 ======================
test_eval_start = time.perf_counter()
ft_tr = evaluate(best_ft_model, train_loader)
ft_va = evaluate(best_ft_model, val_loader)
ft_te = evaluate(best_ft_model, test_loader)
trc_tr = evaluate(ft_trc_model, train_loader, True)
trc_va = evaluate(ft_trc_model, val_loader, True)
trc_te = evaluate(ft_trc_model, test_loader, True)
test_eval_time = time.perf_counter() - test_eval_start
print(f"✅ 全套测试集评估推理耗时：{test_eval_time:.2f} s")

# ====================== 保存FT与FT+TRC对比Excel ======================
res_df = pd.DataFrame([
    {"model":"FT-Transformer",
     "train_acc":ft_tr[0],"train_f1":ft_tr[1],
     "val_acc":ft_va[0],"val_f1":ft_va[1],
     "test_acc":ft_te[0],"test_f1":ft_te[1],
     "train_val_acc_diff":ft_tr[0]-ft_va[0],
     "params":str(best_params),
     },
    {"model":"FT-Transformer+TRC",
     "train_acc":trc_tr[0],"train_f1":trc_tr[1],
     "val_acc":trc_va[0],"val_f1":trc_va[1],
     "test_acc":trc_te[0],"test_f1":trc_te[1],
     "train_val_acc_diff":trc_tr[0]-trc_va[0],
     "params":str(best_params)+"+TRC",
     }
])
res_df.to_excel(r"C:\Users\Qsz53\Desktop\FT_TRC_result.xlsx", index=False)
print("\n✅ 结果已保存：FT_TRC_result-xin.xlsx")
print(res_df.round(4))

# ====================== 保存调参汇总Excel ======================
with pd.ExcelWriter(r"C:\Users\Qsz53\Desktop\FT_TRC_xin调参结果.xlsx", engine="openpyxl") as writer:
    sum_cols = ["d_token","n_blocks","n_heads","lr","attn_drop","ffn_drop","wd",
                "train_acc","train_f1","val_acc","val_f1","train_val_acc_diff","best_epoch","param_train_time_s"]
    summary_df = pd.DataFrame([{k: t[k] for k in sum_cols} for t in all_tuning])
    summary_df.to_excel(writer, sheet_name="参数汇总", index=False)

    for i, t in enumerate(all_tuning):
        pd.DataFrame(t["epoch_history"]).to_excel(
            writer, sheet_name=f"参数{i+1}_epoch", index=False)

    time_df = pd.DataFrame([
        ["FT网格搜索总耗时(s)", round(grid_search_total_time,2)],
        ["FT网格搜索总耗时(min)", round(grid_search_total_time/60,2)],
        ["TRC矫正头训练耗时(s)", round(trc_total_time,2)],
        ["全套模型评估推理耗时(s)", round(test_eval_time,2)],
    ], columns=["项目","耗时"])
    time_df.to_excel(writer, sheet_name="全局实验计时", index=False)

print("\n✅ 每个参数遍历结果已保存：FT_TRC_xin调参结果.xlsx")

# ====================== 混淆矩阵绘图 ======================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_pred_labels(model, loader, is_trc=False):
    model.eval()
    y_true = []
    y_pred = []
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
    return np.array(y_true), np.array(y_pred)

y_true_test, y_pred_ft = get_pred_labels(best_ft_model, test_loader, is_trc=False)
_, y_pred_trc = get_pred_labels(ft_trc_model, test_loader, is_trc=True)
class_names = le.classes_

# FT混淆矩阵
cm_ft = confusion_matrix(y_true_test, y_pred_ft)
print("\n==== FT-Transformer 测试集混淆矩阵 ====")
print(cm_ft)
plt.figure(figsize=(9,7), dpi=300)
sns.heatmap(cm_ft, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("预测岩性")
plt.ylabel("真实岩性")
plt.title("FT-Transformer 测试集混淆矩阵")
plt.tight_layout()
plt.savefig(r"C:\Users\Qsz53\Desktop\FT_xin混淆矩阵.png", bbox_inches="tight")
plt.close()

# FT+TRC混淆矩阵
cm_trc = confusion_matrix(y_true_test, y_pred_trc)
print("\n==== FT-Transformer+TRC 测试集混淆矩阵 ====")
plt.figure(figsize=(9,7), dpi=300)
sns.heatmap(cm_trc, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("预测岩性")
plt.ylabel("真实岩性")
plt.title("FT-Transformer+TRC 测试集混淆矩阵")
plt.tight_layout()
plt.savefig(r"C:\Users\Qsz53\Desktop\FT_TRC_xin混淆矩阵.png", bbox_inches="tight")
plt.close()

print("\n✅ 两张混淆矩阵图片已保存到桌面！")
print("\n🎉 FT+TRC整套实验全部结束！")
