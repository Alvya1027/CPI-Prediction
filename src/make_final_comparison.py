import pandas as pd
import os


# 路径
table_dir = "results/tables"


# ======================
# 1. 基线模型
# ======================

baseline_path = os.path.join(
    table_dir,
    "baseline_results.csv"
)


baseline = pd.read_csv(baseline_path)


baseline_test = baseline.copy()

baseline_test["category"] = "Baseline"



# ======================
# 2. 普通光储备池
# ======================

ordinary_path = os.path.join(
    table_dir,
    "optical_reservoir_metrics.csv"
)


ordinary = pd.read_csv(
    ordinary_path
)


ordinary_test = ordinary[
    ordinary["split"]=="test"
]


ordinary_result = pd.DataFrame({

    "model":["Ordinary Optical Reservoir"],

    "MAE":[
        ordinary_test["mae"].values[0]
    ],

    "RMSE":[
        ordinary_test["rmse"].values[0]
    ],

    "category":[
        "Reservoir"
    ]

})



# ======================
# 3. 孪生光储备池
# ======================

siamese_path=os.path.join(
    table_dir,
    "siamese_optical_reservoir_metrics.csv"
)


siamese=pd.read_csv(
    siamese_path
)


siamese_test=siamese[
    siamese["split"]=="test"
]


siamese_result=pd.DataFrame({

    "model":[
        "Siamese Optical Reservoir"
    ],

    "MAE":[
        siamese_test["cpi_mae"].values[0]
    ],

    "RMSE":[
        siamese_test["cpi_rmse"].values[0]
    ],

    "category":[
        "Siamese Reservoir"
    ]

})



# ======================
# 合并
# ======================


final=pd.concat(
    [
        baseline_test,
        ordinary_result,
        siamese_result
    ],
    ignore_index=True
)


save_path=os.path.join(
    table_dir,
    "final_model_comparison.csv"
)


final.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    "完成:",
    save_path
)

print(final)