# %%
import pandas as pd

df = pd.read_parquet('data/dados_clones.parquet'); df
# %%
features = ["Massa(em kilos)", "Estatura(cm)"]
target = 'Status '
# %%
X = df[features]
y = df[target]
# %%
# categorical to numerical
# no need to convert because the data used is already numerical
# if it were categorical, we would use the following code:
dic_replace = {
  'Tipo 1': 1, 'Tipo 2': 2,
  'Tipo 3': 3, 'Tipo 4': 4,
  'Tipo 5': 5,
  'Yoda': 1, 'Shaak Ti': 2, 'Obi-Wan Kenobi': 3, 'Aayla Secura': 4, 'Mace Windu': 5}
X = X.replace(dic_replace)

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

# %%
from matplotlib import pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model,
                feature_names=features,
                class_names=model.classes_,
                filled=True,
                max_depth=3) # 3 is the maximum depth of the tree