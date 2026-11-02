import pandas as pd
import plotly.graph_objects as go

# ========= 修改为你的文件路径 =========
file_path = r"C:\Users\Qsz53\Desktop\TabPFN3_完整实验报告.xlsx"
# ========================================

# header=1：跳过第1行总标题，第2行作为表头
df = pd.read_excel(file_path, sheet_name="TabPFN3_寻优汇总", header=1)

# 去除列名前后空格
df.columns = df.columns.str.strip()

# F1是纯数字（0~1），直接转浮点，不要除100！
df["train_f1"] = df["train_f1"].astype(float)
df["val_f1"] = df["val_f1"].astype(float)

# 按n_estimators排序
df = df.sort_values("n_estimators").reset_index(drop=True)

fig = go.Figure()

# 左Y轴：train_f1
fig.add_trace(go.Scatter(
    x=df["n_estimators"],
    y=df["train_f1"],
    name="train_f1",
    mode="lines+markers",
    line=dict(width=2.5, color="#3671ad"),
    marker=dict(size=8)
))

# 右Y轴：val_f1
fig.add_trace(go.Scatter(
    x=df["n_estimators"],
    y=df["val_f1"],
    name="val_f1",
    mode="lines+markers",
    line=dict(width=2.5, color="#e8822e"),
    marker=dict(size=8),
    yaxis="y2"
))

fig.update_layout(
    title="",
    title_x=0.5,
    title_font=dict(size=15),
    xaxis=dict(
        title="n_estimators",
        tickmode="linear",
        dtick=1,
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        title="train_f1",
        range=[0.98, 1.0],     # ✅ 修改上限1.0，F1不会出现>1区域
        tickfont=dict(size=11)
    ),
    yaxis2=dict(
        title="val_f1",
        overlaying="y",
        side="right",
        range=[0.78, 0.81],
        tickfont=dict(size=11)
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    width=1000,
    height=560
)

fig.show()
fig.write_html("TabPFN3_双Y折线_F1.html")
