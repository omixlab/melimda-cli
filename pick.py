import pickle

# Abrindo um arquivo .pkl
with open("/home/lucasmocellin/tcc/melimda/Data/Complexos/3d7z/grid_features.pkl", "rb") as f:
    data = pickle.load(f)

# Exibindo o conteúdo
print(data)
