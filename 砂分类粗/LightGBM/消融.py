

import os
import time
import pandas as pd
import numpy as np
import warnings
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

warnings.filterwarnings("ignore")


TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH   = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH  = r"C:\Users\Qsz53\Desktop\测试集.xlsx"

OUT_DIR = r"F:\消融实验\lightgbm"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

num_cols   = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]
cat_cols   = []
label_col  = "岩性"
RANDOM_SEED = 42
early_stop_round = 20


base_fix = {
    "max_depth":4,
    "learning_rate":0.06,
    "n_estimators":150,
    "subsample":0.7,
    "colsample_bytree":0.7
}


exp_setting_list = [
    {
        "exp_id":"Exp1",
        "exp_note":"Ablation_fix_maxdepth=4",
        "fix_param":{"max_depth": base_fix["max_depth"]},
        "search_space":{
            "learning_rate": [0.02, 0.04, 0.06, 0.08],
            "n_estimators": [100, 150, 200],
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9]
        }
    },
    {
        "exp_id":"Exp2",
        "exp_note":"Ablation_fix_lr=0.06",
        "fix_param":{"learning_rate": base_fix["learning_rate"]},
        "search_space":{
            "max_depth": [2, 3, 4, 5, 6],
            "n_estimators": [100, 150, 200],
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9]
        }
    },
    {
        "exp_id":"Exp3",
        "exp_note":"Ablation_fix_nestimators=150",
        "fix_param":{"n_estimators": base_fix["n_estimators"]},
        "search_space":{
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.02, 0.04, 0.06, 0.08],
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9]
        }
    },
    {
        "exp_id":"Exp4",
        "exp_note":"Ablation_fix_subsample=0.7",
        "fix_param":{"subsample": base_fix["subsample"]},
        "search_space":{
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.02, 0.04, 0.06, 0.08],
            "n_estimators": [100, 150, 200],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9]
        }
    },
    {
        "exp_id":"Exp5",
        "exp_note":"Ablation_fix_colsample=0.7",
        "fix_param":{"colsample_bytree": base_fix["colsample_bytree"]},
        "search_space":{
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.02, 0.04, 0.06, 0.08],
            "n_estimators": [100, 150, 200],
            "subsample": [0.6, 0.7, 0.8]
        }
    }
]


df_train = pd.read_excel(TRAIN_PATH)
df_val   = pd.read_excel(VAL_PATH)
df_test  = pd.read_excel(TEST_PATH)


train_wells = set(df_train["井名"].unique())
test_wells = set(df_test["井名"].unique())
overlap_wells = test_wells.intersection(train_wells)
print("="*70)
print(f"训练集井数量：{len(train_wells)}")
print(f"测试集井数量：{len(test_wells)}")
print(f"训练、测试集重叠井：{overlap_wells}")
if len(overlap_wells)>0:
    print("⚠️警告：训练集和测试集存在同一口井！存在数据泄露！")
print("="*70)

X_train = df_train[num_cols + cat_cols].copy()
y_train = df_train[label_col].copy()
X_val = df_val[num_cols + cat_cols].copy()
y_val = df_val[label_col].copy()
X_test = df_test[num_cols + cat_cols].copy()
y_test = df_test[label_col].copy()

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val   = le.transform(y_val)
y_test  = le.transform(y_test)
num_class = len(le.classes_)

time_summary = []


for exp_cfg in exp_setting_list:
    exp_id = exp_cfg["exp_id"]
    exp_note = exp_cfg["exp_note"]
    fix_p = exp_cfg["fix_param"]
    search_p_space = exp_cfg["search_space"]

    print(f"\n>>>>>>>> 开始 {exp_id}: {exp_note}；固定参数：{fix_p}")
    keys_sp = list(search_p_space.keys())
    comb_list = [dict(zip(keys_sp, items)) for items in product(*list(search_p_space.values()))]
    total_comb = len(comb_list)
    print(f"本组待搜索组合数：{total_comb}")

    res_grid_list = []
    best_val_macro_f1 = -1.0
    best_search_params = None

    search_start = time.perf_counter()
    for idx, sp in enumerate(comb_list):
        full_param = {**fix_p, **sp}
        try:
            model = lgb.LGBMClassifier(
                **full_param,
                reg_alpha=0,
                reg_lambda=1,
                random_state=RANDOM_SEED,
                verbose=-1,
                objective="multiclass",
                num_class=num_class,
                class_weight="balanced"
            )
            callbacks = [
                early_stopping(stopping_rounds=early_stop_round),
                log_evaluation(0)
            ]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=callbacks
            )
            y_pred_train = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_pred_train)
            train_f1 = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

            y_pred_val = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
            train_val_acc_diff = train_acc - val_acc

            row = {
                **full_param,
                "train_acc":round(train_acc,4),
                "val_acc":round(val_acc,4),
                "train_f1_macro":round(train_f1,4),
                "val_f1_macro":round(val_f1,4),
                "train_val_acc_diff":round(train_val_acc_diff,4)
            }
            res_grid_list.append(row)

            if val_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_f1
                best_search_params = full_param
        except Exception as e:
            print(f"❌第{idx+1}组训练跳过，错误:{e}")

    search_elapsed = time.perf_counter() - search_start
   
    out_excel = os.path.join(OUT_DIR, f"{exp_id}_{exp_note}_寻优汇总.xlsx")
    pd.DataFrame(res_grid_list).to_excel(out_excel, index=False)

    time_summary.append({
        "exp_id": exp_id,
        "exp_note": exp_note,
        "total_combinations": total_comb,
        "fix_param": str(fix_p),
        "best_val_macro_f1": round(best_val_macro_f1,4),
        "best_params": str(best_search_params),
        "grid_search_time_sec": round(search_elapsed,2),
        "grid_search_time_min": round(search_elapsed/60,2)
    })

    print(f"⏱️ {exp_id}网格搜索耗时：{search_elapsed:.2f}秒（{search_elapsed/60:.2f}分钟）")
    print(f"   最优验证Macro‑F1：{best_val_macro_f1:.4f}；最优参数：{best_search_params}")
    print(f"   ✅保存文件：{out_excel}")


df_time = pd.DataFrame(time_summary)
summary_csv_path = os.path.join(OUT_DIR, "Ablation_Time_Summary.csv")
df_time.to_csv(summary_csv_path, index=False, encoding="utf‑8‑sig")

print("\n" + "="*80)
print("✅LightGBM全部5组消融实验完成！")
print(f"✅总汇总文件路径：{summary_csv_path}")
print(df_time.to_string())
print("="*80)
