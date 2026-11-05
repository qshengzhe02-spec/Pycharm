
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")



TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH   = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH  = r"C:\Users\Qsz53\Desktop\测试集.xlsx"


OUT_DIR = r"F:\消融实验\XGB"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

num_cols  = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]
cat_cols  = ["井名", "层位"]
target_col = "岩性"
RANDOM_SEED = 42
early_stop_round = 20


base_fix = {
    "max_depth": 5,
    "learning_rate": 0.08,
    "n_estimators": 200,
    "subsample": 0.75,
    "colsample_bytree": 0.75
}


exp_setting_list = [
    {
        "exp_id": "Exp1",
        "exp_note": "Ablation_fix_maxdepth=5",
        "fix_param": {"max_depth": base_fix["max_depth"]},
        "search_space": {
            "learning_rate": [0.04, 0.06, 0.08, 0.10],
            "n_estimators": [120, 150, 200, 250],
            "subsample": [0.65, 0.75, 0.85],
            "colsample_bytree": [0.65, 0.75, 0.85]
        }
    },
    {
        "exp_id": "Exp2",
        "exp_note": "Ablation_fix_lr=0.08",
        "fix_param": {"learning_rate": base_fix["learning_rate"]},
        "search_space": {
            "max_depth": [3, 4, 5, 6, 7],
            "n_estimators": [120, 150, 200, 250],
            "subsample": [0.65, 0.75, 0.85],
            "colsample_bytree": [0.65, 0.75, 0.85]
        }
    },
    {
        "exp_id": "Exp3",
        "exp_note": "Ablation_fix_nestimators=200",
        "fix_param": {"n_estimators": base_fix["n_estimators"]},
        "search_space": {
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.04, 0.06, 0.08, 0.10],
            "subsample": [0.65, 0.75, 0.85],
            "colsample_bytree": [0.65, 0.75, 0.85]
        }
    },
    {
        "exp_id": "Exp4",
        "exp_note": "Ablation_fix_subsample=0.75",
        "fix_param": {"subsample": base_fix["subsample"]},
        "search_space": {
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.04, 0.06, 0.08, 0.10],
            "n_estimators": [120, 150, 200, 250],
            "colsample_bytree": [0.65, 0.75, 0.85]
        }
    },
    {
        "exp_id": "Exp5",
        "exp_note": "Ablation_fix_colsample=0.75",
        "fix_param": {"colsample_bytree": base_fix["colsample_bytree"]},
        "search_space": {
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.04, 0.06, 0.08, 0.10],
            "n_estimators": [120, 150, 200, 250],
            "subsample": [0.65, 0.75, 0.85]
        }
    }
]


train_df = pd.read_excel(TRAIN_PATH)
val_df   = pd.read_excel(VAL_PATH)
test_df  = pd.read_excel(TEST_PATH)

le = LabelEncoder()
all_labels = pd.concat([train_df[target_col], val_df[target_col], test_df[target_col]])
le.fit(all_labels)
y_train = le.transform(train_df[target_col])
y_val   = le.transform(val_df[target_col])
y_test  = le.transform(test_df[target_col])

X_train = train_df[num_cols + cat_cols].copy()
X_val   = val_df[num_cols + cat_cols].copy()
X_test  = test_df[num_cols + cat_cols].copy()
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c]   = X_val[c].astype("category")
    X_test[c]  = X_test[c].astype("category")
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
        xgb_clf = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=num_class,
            random_state=RANDOM_SEED,
            enable_categorical=True,
            early_stopping_rounds=early_stop_round,
            verbosity=0,
            **full_param
        )
        xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        ytr_pred = xgb_clf.predict(X_train)
        yv_pred  = xgb_clf.predict(X_val)

        tr_acc = accuracy_score(y_train, ytr_pred)
        v_acc  = accuracy_score(y_val, yv_pred)
        tr_f1  = f1_score(y_train, ytr_pred, average="macro", zero_division=0)
        v_f1   = f1_score(y_val, yv_pred, average="macro", zero_division=0)

        res_grid_list.append({
            **full_param,
            "train_acc": round(tr_acc, 4),
            "val_acc": round(v_acc, 4),
            "train_macro_f1": round(tr_f1, 4),
            "val_macro_f1": round(v_f1, 4),
            "train_val_acc_diff": round(tr_acc - v_acc, 4)
        })
        if v_f1 > best_val_macro_f1:
            best_val_macro_f1 = v_f1
            best_search_params = full_param

    search_elapsed = time.perf_counter() - search_start


    out_excel = os.path.join(OUT_DIR, f"{exp_id}_{exp_note}_寻优汇总.xlsx")
    pd.DataFrame(res_grid_list).to_excel(out_excel, index=False)


    time_summary.append({
        "exp_id": exp_id,
        "exp_note": exp_note,
        "total_combinations": total_comb,
        "fix_param": str(fix_p),
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "best_params": str(best_search_params),
        "grid_search_time_sec": round(search_elapsed, 2),
        "grid_search_time_min": round(search_elapsed / 60, 2)
    })

    print(f"⏱️ {exp_id} 网格搜索耗时：{search_elapsed:.2f} 秒（{search_elapsed/60:.2f} 分钟）")
    print(f"   最优验证Macro‑F1：{best_val_macro_f1:.4f}；最优参数：{best_search_params}")
    print(f"   ✅寻优汇总已保存：{out_excel}")


df_time = pd.DataFrame(time_summary)
summary_csv_path = os.path.join(OUT_DIR, "Ablation_Time_Summary.csv")
df_time.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("✅ 5组消融实验全部完成！")
print(f"✅ 总耗时汇总文件：{summary_csv_path}")
print(df_time.to_string())
print("=" * 70)
