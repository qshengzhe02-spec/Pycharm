
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
import xgboost as xgb
import warnings
from itertools import product

warnings.filterwarnings("ignore")


TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH   = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH  = r"C:\Users\Qsz53\Desktop\测试集.xlsx"

num_cols  = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]
cat_cols  = ["井名", "层位"]
target_col = "岩性"
RANDOM_SEED = 42


param_space = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.04, 0.06, 0.08, 0.10],
    "n_estimators": [120, 150, 200, 250],
    "subsample": [0.65, 0.75, 0.85],
    "colsample_bytree": [0.65, 0.75, 0.85]
}
early_stop_round = 20


keys = list(param_space.keys())
values_list = list(param_space.values())
all_param_combinations = [dict(zip(keys, items)) for items in product(*values_list)]
print(f"✅扩充网格，总待运行参数组合数量：{len(all_param_combinations)}")



train_df = pd.read_excel(TRAIN_PATH)
val_df   = pd.read_excel(VAL_PATH)
test_df  = pd.read_excel(TEST_PATH)


train_wells = set(train_df["井名"].unique())
test_wells = set(test_df["井名"].unique())
overlap_wells = test_wells.intersection(train_wells)
print("="*60)
print(f"训练集井数量：{len(train_wells)}")
print(f"测试集井数量：{len(test_wells)}")
print(f"训练、测试集重叠井：{overlap_wells}")
if len(overlap_wells) > 0:
    print("⚠️警告：训练集和测试集存在同一口井！存在数据泄露，测试集指标虚高，新井泛化差！")
print("="*60)


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


result_list = []
best_val_macro_f1 = -1.0
best_model = None
best_params = None

search_start = time.perf_counter()
for idx, params in enumerate(all_param_combinations):
    print(f"\n========== 第{idx+1}/{len(all_param_combinations)}组参数 ==========")
    print(params)
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_class,
        random_state=RANDOM_SEED,
        enable_categorical=True,
        early_stopping_rounds=early_stop_round,
        verbosity=0,
        **params
    )
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    y_train_pred = xgb_clf.predict(X_train)
    y_val_pred   = xgb_clf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc   = accuracy_score(y_val, y_val_pred)
    train_f1  = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
    val_f1    = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    train_val_acc_diff = train_acc - val_acc

    result_list.append({
        **params,
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "train_macro_f1": round(train_f1, 4),
        "val_macro_f1": round(val_f1, 4),
        "train_val_acc_diff": round(train_val_acc_diff, 4)
    })
    print(f"train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}")
    print(f"train_macro_f1:{train_f1:.4f}, val_macro_f1:{val_f1:.4f}, acc_diff:{train_val_acc_diff:.4f}")

    if val_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_f1
        best_model = xgb_clf
        best_params = params

search_elapsed = time.perf_counter() - search_start
print("\n" + "="*60)
print(f"⏱️ 扩充网格参数搜索总耗时：{search_elapsed:.2f} 秒 "
      f"（约 {search_elapsed/60:.2f} 分钟），共 {len(all_param_combinations)} 组")
print("="*60)


df_grid_result = pd.DataFrame(result_list)
df_grid_result.to_excel(r"C:\Users\Qsz53\Desktop\XGBoost_ExpandGrid_720组寻优汇总.xlsx", index=False)
print("\n✅720组扩充网格搜索结果保存：XGBoost_ExpandGrid_720组寻优汇总.xlsx")
print(f"✅最优参数组合：{best_params}")
print(f"✅最优验证集Macro-F1: {best_val_macro_f1:.4f}")


print("\n" + "="*50)
print("==== 最优XGBoost模型【测试集】评价 ====")

test_start = time.perf_counter()
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
test_elapsed = time.perf_counter() - test_start

print(f"Accuracy  : {acc:.4f}")
print(f"Macro-F1  : {macro_f1:.4f}")
print(f"⏱️ 最优参数模型测试集评估耗时：{test_elapsed:.4f} 秒")
print("="*50)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("XGBoost_ExpandGrid Best Model Confusion Matrix")
plt.xlabel("预测岩性")
plt.ylabel("真实岩性")
plt.tight_layout()
plt.savefig(r"C:\Users\Qsz53\Desktop\xgb_expandgrid_best_confusion_matrix.png", dpi=200)
plt.close()


test_df["pred_xgb_expandgrid_best"] = le.inverse_transform(y_pred)
test_df.to_csv(r"C:\Users\Qsz53\Desktop\xgb_expandgrid_best_test_pred.csv",
               index=False, encoding="utf-8-sig")


res_df = pd.DataFrame([{
    "model": "XGBoost_ExpandGrid_720",
    "acc": acc,
    "macro_f1": macro_f1,
    "best_params": str(best_params),
    "grid_search_time_sec": round(search_elapsed, 2),
    "grid_search_time_min": round(search_elapsed / 60, 2),
    "test_eval_time_sec": round(test_elapsed, 4)
}])
res_df.to_csv(r"C:\Users\Qsz53\Desktop\xgb_expandgrid_best_result.csv",
              index=False, encoding="utf-8-sig")


plt.figure(figsize=(10, 6))
feature_names = list(X_train.columns)
fi = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
fi.plot(kind="bar")
plt.title("XGBoost_ExpandGrid 特征重要性(最优模型)")
plt.ylabel("importance")
plt.tight_layout()
plt.savefig(r"C:\Users\Qsz53\Desktop\xgb_expandgrid_feature_importance.png", dpi=200)
plt.close()

print("\n✅【XGBoost扩充720组网格版本】输出文件清单：")
print("1. XGBoost_ExpandGrid_720组寻优汇总.xlsx   【全部720组参数记录】")
print("2. xgb_expandgrid_best_confusion_matrix.png 【混淆矩阵】")
print("3. xgb_expandgrid_feature_importance.png    【特征重要性】")
print("4. xgb_expandgrid_best_test_pred.csv        【测试集预测】")
print("5. xgb_expandgrid_best_result.csv           【指标+两段耗时】")
print("\n⏱️ 汇总：")
print(f"   参数搜索耗时：{search_elapsed:.2f} 秒（约 {search_elapsed/60:.2f} 分钟）")
print(f"   测试集评估耗时：{test_elapsed:.4f} 秒")
