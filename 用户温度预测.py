
import numpy as np
import pandas as pd
import joblib
df = pd.read_csv("插补后验证集.csv").fillna(0)
def clr_transform(data):
    data = data.replace(0, 1e-8)
    geometric_mean = np.exp(np.log(data).mean(axis=1)) 
    clr_data = np.log(data.div(geometric_mean, axis=0))
    return clr_data
selected_features = [
    "SiO2.cpx", "TiO2.cpx", "Al2O3.cpx", "Cr2O3.cpx", "FeOt.cpx", "MgO.cpx",
    "MnO.cpx", "CaO.cpx", "Na2O.cpx",
    "SiO2.liq", "TiO2.liq", "Al2O3.liq", "FeOt.liq", "MgO.liq", "MnO.liq",
    "CaO.liq", "Na2O.liq", "K2O.liq", "P2O5.liq"
]
model_features = [
    "FeOt.cpx", "CaO.liq", "P2O5.liq", "K2O.liq", "Na2O.cpx", "MnO.liq",
    "FeOt.liq", "MnO.cpx", "Al2O3.cpx", "Al2O3.liq", "MgO.liq",
    "TiO2.cpx", "CaO.cpx", "TiO2.liq", "SiO2.liq"
]
clr_df = clr_transform(df[selected_features])
clr_model_input = clr_df[model_features]
model = joblib.load('最终温度预测模型.pkl')
all_tree_predictions = np.array([tree.predict(clr_model_input) for tree in model.estimators_])
pred_mean = np.mean(all_tree_predictions, axis=0)
pred_iqr = np.percentile(all_tree_predictions, 75, axis=0) - np.percentile(all_tree_predictions, 25, axis=0)
output = df.copy()
output["Predicted_Temp"] = pred_mean
output["Temp_IQR"] = pred_iqr
output.to_csv("output.csv", index=False)


