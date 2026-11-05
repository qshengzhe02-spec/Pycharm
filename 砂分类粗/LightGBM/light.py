
"""
import time
import pandas as pd
import numpy as np
import warnings
from itertools import product

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH   = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH  = r"C:\Users\Qsz53\Desktop\测试集.xlsx"

num_cols   = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]
cat_cols   = []
label_col  = "岩性"
RANDOM_SEED = 42
early_stop_round = 20


param_space = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.02, 0.04, 0.06, 0.08],
    "n_estimators": [100, 150, 200],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9]
}

for k,v in param_space.items():
    print(f"{k} 候选列表：{v}，数量={len(v)}")



keys = list(param_space.keys())
values_list = list(param_space.values())
all_param_combinations = [dict(zip(keys, items)) for items in product(*values_list)]
print(f"✅LightGBM完整网格，总待运行参数组合数量：{len(all_param_combinations)}")


# all_param_combinations = all_param_combinations[:20]


df_train = pd.read_excel(TRAIN_PATH)
df_val   = pd.read_excel(VAL_PATH)
df_test  = pd.read_excel(TEST_PATH)


train_wells = set(df_train["井名"].unique())
test_wells = set(df_test["井名"].unique())
overlap_wells = test_wells.intersection(train_wells)
print("="*60)
print(f"训练集井数量：{len(train_wells)}")
print(f"测试集井数量：{len(test_wells)}")
print(f"训练、测试集重叠井：{overlap_wells}")
if len(overlap_wells)>0:
    print("⚠️警告：训练集和测试集存在同一口井！存在数据泄露，测试集指标虚高！")
print("="*60)

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


best_val_f1 = -1.0
best_model = None
best_params = None
result_records = []

search_start = time.perf_counter()

for idx, p in enumerate(all_param_combinations):
    print(f"\n========== 第{idx+1}/{len(all_param_combinations)}组参数 ==========")
    print(p)
    try:
        model = lgb.LGBMClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
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
            "model":"LightGBM_balanced",
            **p,
            "train_acc":round(train_acc,4),
            "val_acc":round(val_acc,4),
            "train_f1_macro":round(train_f1,4),
            "val_f1_macro":round(val_f1,4),
            "train_val_acc_diff":round(train_val_acc_diff,4)
        }
        result_records.append(row)
        print(f"val_f1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = p
    except Exception as e:
        print(f"❌第{idx+1}组参数训练失败，跳过！错误信息：{e}")

search_elapsed = time.perf_counter() - search_start
print("\n" + "="*60)
print(f"⏱️ LightGBM网格搜索总耗时：{search_elapsed:.2f} 秒（约 {search_elapsed/60:.2f} 分钟）")
print("="*60)


print("\n=====最优模型测试集结果=====")
test_start = time.perf_counter()
y_pred_test = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
test_elapsed = time.perf_counter() - test_start

print(f"最优参数：{best_params}")
print(f"Test Acc: {test_acc:.4f}, Test Macro‑F1:{test_f1:.4f}")
print(f"⏱️ 测试集评估耗时：{test_elapsed:.4f} 秒")
print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))


res_df = pd.DataFrame(result_records)

best_train_pred = best_model.predict(X_train)
best_val_pred = best_model.predict(X_val)
res_df.loc[len(res_df)] = {
    "model":"LightGBM_balanced_BEST",
    **best_params,
    "train_acc":round(accuracy_score(y_train, best_train_pred),4),
    "val_acc":round(accuracy_score(y_val, best_val_pred),4),
    "train_f1_macro":round(f1_score(y_train, best_train_pred, average="macro", zero_division=0),4),
    "val_f1_macro":round(best_val_f1,4),
    "train_val_acc_diff":round(accuracy_score(y_train,best_train_pred) - accuracy_score(y_val,best_val_pred),4),
}
res_df["test_acc"] = round(test_acc,4)
res_df["test_f1_macro"] = round(test_f1,4)

res_df["grid_search_time_sec"] = round(search_elapsed,2)
res_df["grid_search_time_min"] = round(search_elapsed/60,2)
res_df["test_eval_time_sec"] = round(test_elapsed,4)


cm = confusion_matrix(y_test, y_pred_test)
class_names = le.classes_


print("\n==== 测试集混淆矩阵（文本） ====")
print(cm)


plt.figure(figsize=(9, 7), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("预测岩性")
plt.ylabel("真实岩性")
plt.title(f"LightGBM_720Grid最优模型测试集混淆矩阵({best_params})")
plt.tight_layout()
plt.savefig(r"C:\Users\Qsz53\Desktop\LightGBM_720Grid_混淆矩阵.png", bbox_inches="tight")
plt.close()
print("✅ 混淆矩阵图片已保存：LightGBM_720Grid_混淆矩阵.png")

res_df.to_excel(r"C:\Users\Qsz53\Desktop\LightGBM_720Grid_完整寻优汇总.xlsx", index=False)
print("\n✅LightGBM 720组网格结果保存到桌面：LightGBM_720Grid_完整寻优汇总.xlsx")

print("\n✅输出文件清单：")
print("1. LightGBM_720Grid_完整寻优汇总.xlsx  全部网格记录+耗时")
print("2. LightGBM_720Grid_混淆矩阵.png      最优模型混淆矩阵图")
print("\n⏱️汇总：")
print(f"   网格搜索耗时：{search_elapsed:.2f}秒（约 {search_elapsed/60:.2f}分钟）")
print(f"   测试集评估耗时：{test_elapsed:.4f}秒")
