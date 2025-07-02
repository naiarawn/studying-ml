# %%

import pandas as pd

# %%

df = pd.read_excel('data/dados_frutas.xlsx')
df

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%
y = df['Fruta']
caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
X = df[caracteristicas]

# %%
# cria a árvore de decisão
# X = df[caracteristicas] -> seleciona as colunas de características
# y = df['Fruta'] -> seleciona a coluna de rótulos
# arvore.fit(X, y) -> treina a árvore de decisão com as características e rótulos
arvore.fit(X, y)
# %%
arvore.predict([[1, 1, 1, 1]]) #cereja
arvore.predict([[0, 1, 1, 1]]) #morango
arvore.predict([[0, 0, 0, 0]]) #banana

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_, #pegar os rotulos únicos e ordená-los
               filled=True)

# %%
# probabilidade de cada fruta a partir das características
proba = arvore.predict_proba([[1, 1, 1, 1]])[0]
pd.Series(proba, index=arvore.classes_)