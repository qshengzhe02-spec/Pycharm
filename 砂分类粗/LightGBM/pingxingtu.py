import pandas as pd
import plotly.express as px

# 读取你的LightGBM寻优表格，修改为你自己的文件路径
df = pd.read_excel(r"C:\Users\Qsz53\Desktop\LightGBM_720Grid_完整寻优汇总.xlsx")

dim_list = [
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "train_f1_macro",
    "val_f1_macro"
]

fig = px.parallel_coordinates(
    df[dim_list],
    color="val_f1_macro",
    color_continuous_scale=["#234b8c", "#3671ad", "#4c98c6", "#65b9b0", "#92d28d", "#e4c348", "#ec802e", "#ca2e2c"],
    dimensions=dim_list,
    title="",
    range_color=[0.28, 0.48]
)

fig.update_layout(
    width=1400,
    height=680,
    title_x=0.5,
    title_font={"size":16},
    font={"size":11},
    coloraxis_colorbar=dict(
        title="val_f1_macro",
        title_font={"size":12},
        tickfont={"size":10},
        len=0.78
    )
)

# 设置坐标轴字体，修复之前报错
fig.update_traces(
    labelfont=dict(size=11),
    tickfont=dict(size=10),
    selector=dict(type='parcoords')
)

# 配置网页导出参数 scale=3 三倍高清PNG
config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename":"LightGBM_平行坐标图",
        "width":1400,
        "height":680,
        "scale":3
    }
}

fig.show(config=config)
# kaleido经常浏览器崩溃，注释掉write_image，改用网页相机按钮导出
# fig.write_image("LightGBM_平行坐标图.png", width=1400, height=680)
fig.write_html("LightGBM_平行坐标图.html", config=config)
