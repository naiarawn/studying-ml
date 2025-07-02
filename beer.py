# %%
import pandas as pd

df = df = pd.read_excel('data/dados_cerveja.xlsx')
df.head()

# %%
features = ["temperatura", "copo", "espuma", "cor"]
target = 'classe'

X = df[features]
y = df[target]

#categorical to numerical
X = X.replace({
  'mud': 1, 'pint': 2,
  'sim': 1, 'não': 0,
  'clara': 0, 'escura': 1
})

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model,
                feature_names=features,
               class_names=model.classes_,
               filled=True)

# %%
model.predict([[-5, 2, 1, 0]])[0]
proba = model.predict_proba([[-5, 2, 1, 0]])[0]
pd.Series(proba, index=model.classes_)