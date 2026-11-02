# ====================== 必须放在最前面！解除CPU大数据样本限制 ======================
import os
import openpyxl
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import warnings
warnings.filterwarnings("ignore")

import time  # ★ 新增：计时模块
import numpy as np
import pandas as pd
import joblib
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tabpfn import TabPFNClassifier

# ====================== 1. 配置区（按你的实际情况修改） ======================
TRAIN_PATH = r"C:\Users\Qsz53\Desktop\训练集.xlsx"
VAL_PATH = r"C:\Users\Qsz53\Desktop\验证集.xlsx"
TEST_PATH = r"C:\Users\Qsz53\Desktop\测试集.xlsx"

# 特征列与标签列
num_cols = ["GR", "CAL", "SP", "AC", "DEN", "CNL", "RD", "U", "K"]  # 数值测井曲线
cat_cols = ["井名", "层位"]  # 分类特征
target_col = "岩性"  # 预测目标

# ★ 修改点1：n_estimators候选值增加
N_ESTIMATORS_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 固定随机种子，保证实验可复现
RANDOM_STATE = 42
# 设备：有NVIDIA显卡可改为"cuda"加速
DEVICE = "cpu"

# ====================== 2. 通用工具函数 ======================
def display_width(text):
    if text is None:
        return 0
    s = str(text)
    if not s:
        return 0
    lines = s.split('\n')
    def line_w(line):
        w = 0
        for ch in line:
            w += 2 if ord(ch) > 127 else 1
        return w
    return max(line_w(l) for l in lines)

def to_native(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v

def auto_fit_columns(ws, min_w=10, max_w=50, padding=3, skip_cols=()):
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        if letter in skip_cols:
            continue
        col_cells = ws[letter]
        w = max(
            (display_width(c.value) for c in col_cells
             if not isinstance(c, openpyxl.cell.cell.MergedCell) and c.value is not None),
            default=0,
        )
        ws.column_dimensions[letter].width = max(min_w, min(w * 1.1 + padding, max_w))

def build_excel_sheet(wb, sheet_name, headers, rows, title,
                      pct_cols=(), int_cols=(), highlight_best_col=None):
    import openpyxl
    ws = wb.create_sheet(sheet_name) if sheet_name in wb.sheetnames or wb.sheetnames != ['Sheet'] else wb.active
    if ws.title != sheet_name:
        ws.title = sheet_name

    n_cols = len(headers)
    last_col_letter = get_column_letter(n_cols)

    title_font = Font(name='微软雅黑', bold=True, color='FFFFFF', size=13)
    title_fill = PatternFill('solid', fgColor='1F4E78')
    header_font = Font(name='微软雅黑', bold=True, color='FFFFFF', size=11)
    header_fill = PatternFill('solid', fgColor='0070C0')
    data_font = Font(name='微软雅黑', size=10)
    best_fill = PatternFill('solid', fgColor='FFF2CC')
    best_font = Font(name='微软雅黑', size=10, bold=True, color='9C6500')

    thin = Side(style='thin', color='BFBFBF')
    full_border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)
    left = Alignment(horizontal='left', vertical='center', wrap_text=True)
    right = Alignment(horizontal='right', vertical='center')

    ws.merge_cells(f"A1:{last_col_letter}1")
    tcell = ws.cell(row=1, column=1, value=title)
    tcell.font = title_font
    tcell.fill = title_fill
    tcell.alignment = center
    for col_idx in range(1, n_cols + 1):
        ws.cell(row=1, column=col_idx).border = full_border
    ws.row_dimensions[1].height = 28

    header_row = 2
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=header_row, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = full_border
    ws.row_dimensions[header_row].height = 24

    data_start = header_row + 1
    best_row_idx = None
    if highlight_best_col and highlight_best_col in headers and rows:
        col_pos = headers.index(highlight_best_col)
        values = [r[col_pos] for r in rows]
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if valid:
            best_val = max(v for _, v in valid)
            best_row_idx = [i for i, v in valid if v == best_val][0]

    for r_offset, row_data in enumerate(rows):
        excel_row = data_start + r_offset
        is_best = (r_offset == best_row_idx)
        zebra_fill = PatternFill('solid', fgColor='F2F7FC') if r_offset % 2 == 0 else PatternFill('solid', fgColor='FFFFFF')

        for col_idx, raw_val in enumerate(row_data, 1):
            header_name = headers[col_idx - 1]
            val = to_native(raw_val)
            cell = ws.cell(row=excel_row, column=col_idx, value=val)

            if is_best:
                cell.font = best_font
                cell.fill = best_fill
            else:
                cell.font = data_font
                cell.fill = zebra_fill
            cell.border = full_border

            if header_name in pct_cols and isinstance(val, (int, float)):
                cell.number_format = '0.00%'
                cell.alignment = right
            elif header_name in int_cols and isinstance(val, (int, float)):
                cell.number_format = '0'
                cell.alignment = right
            elif isinstance(val, (int, float)):
                cell.alignment = right
            else:
                if isinstance(val, str) and display_width(val) > 12:
                    cell.alignment = left
                else:
                    cell.alignment = center

        ws.row_dimensions[excel_row].height = 20

    ws.freeze_panes = f"A{data_start}"
    last_row = data_start + len(rows) - 1 if rows else header_row
    ws.auto_filter.ref = f"A{header_row}:{last_col_letter}{last_row}"
    auto_fit_columns(ws, min_w=12, max_w=42, padding=3)
    return ws

# ====================== 3. 数据读取与校验 ======================
def load_and_validate(path, required_cols, df_name):
    df = pd.read_excel(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{df_name}] 缺少必需列: {missing}，实际列: {list(df.columns)}")
    print(f"[{df_name}] 读取成功，形状: {df.shape}")
    return df

required_all = num_cols + cat_cols + [target_col]
train_df = load_and_validate(TRAIN_PATH, required_all, "训练集")
val_df = load_and_validate(VAL_PATH, required_all, "验证集")
test_df = load_and_validate(TEST_PATH, required_all, "测试集")

# ====================== 4. 缺失值处理 ======================
def fill_missing(df, num_cols, cat_cols):
    df = df.copy()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("未知").astype(str)
    return df

train_df = fill_missing(train_df, num_cols, cat_cols)
val_df = fill_missing(val_df, num_cols, cat_cols)
test_df = fill_missing(test_df, num_cols, cat_cols)

# ====================== 5. 标签统一编码 ======================
le = LabelEncoder()
all_labels = pd.concat([train_df[target_col], val_df[target_col], test_df[target_col]])
le.fit(all_labels)

y_train = le.transform(train_df[target_col])
y_val = le.transform(val_df[target_col])
y_test = le.transform(test_df[target_col])

# ====================== 6. 特征预处理（严格防止数据泄露） ======================
scaler = StandardScaler()
oe = OrdinalEncoder()

X_train_num_s = scaler.fit_transform(train_df[num_cols])
X_train_cat_s = oe.fit_transform(train_df[cat_cols]) if cat_cols else None
X_val_num_s = scaler.transform(val_df[num_cols])
X_val_cat_s = oe.transform(val_df[cat_cols]) if cat_cols else None
X_test_num_s = scaler.transform(test_df[num_cols])
X_test_cat_s = oe.transform(test_df[cat_cols]) if cat_cols else None

if cat_cols:
    X_train = np.concatenate([X_train_num_s, X_train_cat_s], axis=1)
    X_val = np.concatenate([X_val_num_s, X_val_cat_s], axis=1)
    X_test = np.concatenate([X_test_num_s, X_test_cat_s], axis=1)
else:
    X_train = X_train_num_s
    X_val = X_val_num_s
    X_test = X_test_num_s

print(f"\n特征形状 → 训练集:{X_train.shape} | 验证集:{X_val.shape} | 测试集:{X_test.shape}")

# ====================== 7. 自动寻优：遍历n_estimators【★计时开始】 ======================
print("\n===== 开始自动寻优 =====")
print(f"候选参数: n_estimators = {N_ESTIMATORS_CANDIDATES}")

tuning_results = []
search_start = time.perf_counter()   # ★ 计时起点：自动寻优

for n in N_ESTIMATORS_CANDIDATES:
    print(f"\n正在训练 n_estimators = {n} ...")
    clf = TabPFNClassifier(
        device=DEVICE,
        n_estimators=n,
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_train_pred = clf.predict(X_train)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)

    tuning_results.append({
        "n_estimators": n,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "train_val_acc_diff": train_acc - val_acc
    })
    print(f"n={n} | 训练Acc: {train_acc:.4f} | 验证Acc: {val_acc:.4f} | 验证Macro-F1: {val_f1:.4f}")

search_elapsed = time.perf_counter() - search_start   # ★ 计时终点：自动寻优

tuning_df = pd.DataFrame(tuning_results).sort_values("val_f1", ascending=False).reset_index(drop=True)
print("\n===== 寻优结果汇总（按验证集Macro-F1降序） =====")
print(tuning_df.round(4))

best_params = tuning_df.iloc[0].to_dict()
best_n = int(best_params["n_estimators"])
print(
    f"\n✅ 最优参数: n_estimators = {best_n} | 验证集Acc: {best_params['val_acc']:.4f} | 验证集Macro-F1: {best_params['val_f1']:.4f}")
print(f"⏱️ 自动寻优（{len(N_ESTIMATORS_CANDIDATES)}个候选）总耗时：{search_elapsed:.2f} 秒（约 {search_elapsed/60:.2f} 分钟）")

# ====================== 8. 用最优参数训练最终模型，测试集评估【★计时】 ======================
print("\n===== 最终模型训练与测试集评估 =====")
final_clf = TabPFNClassifier(
    device=DEVICE,
    n_estimators=best_n,
    random_state=RANDOM_STATE
)
final_clf.fit(X_train, y_train)

y_train_pred_final = final_clf.predict(X_train)
y_val_pred_final = final_clf.predict(X_val)

test_start = time.perf_counter()   # ★ 计时起点：测试集评估
y_test_pred_final = final_clf.predict(X_test)
test_elapsed = time.perf_counter() - test_start   # ★ 计时终点：测试集评估

def calc_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n==== {name}集评价 ====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1_macro:.4f}")
    return acc, f1_macro

calc_metrics(y_train, y_train_pred_final, "训练")
calc_metrics(y_val, y_val_pred_final, "验证")
calc_metrics(y_test, y_test_pred_final, "测试")
print(f"⏱️ 最优模型测试集评估耗时：{test_elapsed:.4f} 秒")

print("\n==== 测试集分类报告 ====")
print(classification_report(y_test, y_test_pred_final, target_names=le.classes_, zero_division=0))

print("\n==== 测试集混淆矩阵 ====")
print(confusion_matrix(y_test, y_test_pred_final))
# ===================== 绘制混淆矩阵图 =====================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cm = confusion_matrix(y_test, y_test_pred_final)
class_names = le.classes_

plt.figure(figsize=(8,6), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("预测岩性")
plt.ylabel("真实岩性")
plt.title(f"TabPFN-2最优模型测试集混淆矩阵(n_estimators={best_n})")
plt.tight_layout()
plt.savefig(r"C:\Users\Qsz53\Desktop\TabPFN-2_混淆矩阵.png", bbox_inches="tight")
plt.close()
print("\n✅ 混淆矩阵图片已保存到桌面：TabPFN-2_混淆矩阵.png")

# ====================== 9. 保存模型与预测结果 ======================
joblib.dump(final_clf, "TabPFN_最优岩性识别模型.pkl")
print("\n✅ 模型已保存为: TabPFN_最优岩性识别模型.pkl")

test_result_df = test_df.copy()
test_result_df["预测岩性"] = le.inverse_transform(y_test_pred_final)
test_result_df["预测正确"] = test_result_df["岩性"] == test_result_df["预测岩性"]
test_result_df.to_excel("测试集_预测结果.xlsx", index=False)
print("✅ 测试集预测结果已保存为: 测试集_预测结果.xlsx")

# ====================== 10. 保存寻优结果Excel（带格式） ======================
wb = Workbook()

# --- Sheet 1: 寻优结果汇总 ---
tuning_headers = ["n_estimators", "train_acc", "train_f1", "val_acc", "val_f1", "train_val_acc_diff"]
tuning_rows = [
    [
        int(r["n_estimators"]),
        float(r["train_acc"]),
        float(r["train_f1"]),
        float(r["val_acc"]),
        float(r["val_f1"]),
        float(r["train_val_acc_diff"]),
    ]
    for r in tuning_results
]
build_excel_sheet(
    wb,
    sheet_name="寻优结果汇总",
    headers=tuning_headers,
    rows=tuning_rows,
    title="TabPFN n_estimators 自动寻优结果",
    pct_cols={"train_acc", "train_f1", "val_acc", "val_f1", "train_val_acc_diff"},
    int_cols={"n_estimators"},
    highlight_best_col="val_f1",
)

# --- Sheet 2: 最优模型测试集评估 ---
cr = classification_report(
    y_test, y_test_pred_final,
    target_names=le.classes_, zero_division=0,
    output_dict=True,
)
cr_headers = ["岩性类别", "precision", "recall", "f1-score", "support"]
cr_rows = []
for class_name in le.classes_:
    row = cr.get(class_name)
    if row is None:
        cr_rows.append([class_name, None, None, None, None])
    else:
        cr_rows.append([
            class_name,
            float(row["precision"]),
            float(row["recall"]),
            float(row["f1-score"]),
            int(row["support"]),
        ])
cr_rows.append([
    "macro avg",
    float(cr["macro avg"]["precision"]),
    float(cr["macro avg"]["recall"]),
    float(cr["macro avg"]["f1-score"]),
    int(cr["macro avg"]["support"]),
])
cr_rows.append([
    "weighted avg",
    float(cr["weighted avg"]["precision"]),
    float(cr["weighted avg"]["recall"]),
    float(cr["weighted avg"]["f1-score"]),
    int(cr["weighted avg"]["support"]),
])
build_excel_sheet(
    wb,
    sheet_name="最优模型分类报告",
    headers=cr_headers,
    rows=cr_rows,
    title=f"最优模型(n_estimators={best_n}) 测试集分类报告",
    pct_cols={"precision", "recall", "f1-score"},
    int_cols={"support"},
)

# --- Sheet 3: 混淆矩阵 ---
cm = confusion_matrix(y_test, y_test_pred_final)
cm_headers = ["真实岩性"] + [str(c) for c in le.classes_]
cm_rows = []
for i, true_class in enumerate(le.classes_):
    row = [true_class] + [int(x) for x in cm[i]]
    cm_rows.append(row)
build_excel_sheet(
    wb,
    sheet_name="测试集混淆矩阵",
    headers=cm_headers,
    rows=cm_rows,
    title=f"最优模型(n_estimators={best_n}) 测试集混淆矩阵",
    int_cols=set(le.classes_),
)

# --- Sheet 4: 核心指标汇总【★ 新增耗时列】 ---
train_metrics = calc_metrics(y_train, y_train_pred_final, "训练")
val_metrics = calc_metrics(y_val, y_val_pred_final, "验证")
test_metrics = calc_metrics(y_test, y_test_pred_final, "测试")
summary_headers = ["数据集", "Accuracy", "Macro-F1"]
summary_rows = [
    ["训练集", float(train_metrics[0]), float(train_metrics[1])],
    ["验证集", float(val_metrics[0]), float(val_metrics[1])],
    ["测试集", float(test_metrics[0]), float(test_metrics[1])],
]
build_excel_sheet(
    wb,
    sheet_name="核心指标汇总",
    headers=summary_headers,
    rows=summary_rows,
    title=f"最优模型(n_estimators={best_n}) 核心指标汇总",
    pct_cols={"Accuracy", "Macro-F1"},
)

# --- Sheet 5: 实验耗时记录【★ 新增Sheet】 ---
time_headers = ["项目", "耗时(秒)", "耗时(分钟)"]
time_rows = [
    ["自动寻优（n_estimators遍历）", round(search_elapsed, 2), round(search_elapsed / 60, 2)],
    ["最优模型测试集评估", round(test_elapsed, 4), round(test_elapsed / 60, 4)],
]
build_excel_sheet(
    wb,
    sheet_name="实验耗时",
    headers=time_headers,
    rows=time_rows,
    title="TabPFN 实验计时记录",
)

# 保存Excel
wb.save("TabPFN_自动寻优结果.xlsx")
print("✅ 寻优结果Excel已保存为: TabPFN_自动寻优结果.xlsx")
print("\n🎉 全部流程完成！")
print(f"⏱️ 汇总：自动寻优耗时 {search_elapsed:.2f}秒；测试集评估耗时 {test_elapsed:.4f}秒")
