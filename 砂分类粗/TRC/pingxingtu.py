import pandas as pd
import plotly.express as px

# ========= 修改文件路径 =========
file_path = r"C:\Users\Qsz53\Desktop\FT_TRC_xin调参结果.xlsx"
# ================================

df = pd.read_excel(file_path, sheet_name="参数汇总")
print("Excel列名：", df.columns.tolist())

dim_list = [
    "d_token",
    "n_blocks",
    "n_heads",
    "lr",
    "attn_drop",
    "ffn_drop",
    "wd",
    "train_f1",
    "val_f1"
]

# 使用plotly自带色板Viridis，避免自定义色阶格式报错
fig = px.parallel_coordinates(
    df[dim_list],
    color="val_f1",
    color_continuous_scale="Viridis",
    dimensions=dim_list,
    title="",
    range_color=[0.22, 0.31]
)

# 仅布局设置，删掉所有update_traces！！
fig.update_layout(
    width=1500,
    height=700,
    title_x=0.5,
    title_font=dict(size=16),
    font=dict(size=11),
    coloraxis_colorbar=dict(
        title="val_f1",
        title_font=dict(size=12),
        tickfont=dict(size=10),
        len=0.78
    )
)

# 导出图片配置
config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename":"FT_TRC_xin平行坐标图",
        "width":1500,
        "height":700,
        "scale":3
    }
}

fig.show(config=config)
fig.write_html("FT_TRC_xin平行坐标图.html", config=config)
