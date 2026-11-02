import pandas as pd
import plotly.express as px

df = pd.read_excel(r"C:\Users\Qsz53\Desktop\XGBoost_ExpandGrid_720组寻优汇总.xlsx")

dim_list = [
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "train_macro_f1",
    "val_macro_f1"
]

fig = px.parallel_coordinates(
    df[dim_list],
    color="val_macro_f1",
    color_continuous_scale=["#234b8c", "#3878c2", "#59a8d0", "#78c9bc", "#eebb44", "#f08838", "#dd3b32"],
    dimensions=dim_list,
    title="",
    range_color=[0.52, 0.74]
)

fig.update_layout(
    width=1400,
    height=650,
    title_x=0.5,
    title_font={"size":16},
    font={"size":11},
    coloraxis_colorbar=dict(
        title="val_macro_f1",
        title_font={"size":12},
        tickfont={"size":10},
        len=0.75
    )
)

# ✅ 正确：labelfont 和 tickfont 放在update_traces顶层，不是dimensions内部
fig.update_traces(
    labelfont=dict(size=11),
    tickfont=dict(size=10),
    selector=dict(type='parcoords')
)

fig.show()
# fig.write_image("parallel_7dim_f1.svg", width=1400, height=650)
fig.write_html("parallel_7dim_f1.html")
