import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregar o dataset (substitua 'seu_dataset.csv' pelo nome do seu arquivo)
dataset = pd.read_csv('dataset.csv')

# Escolher os campos para a entrada do modelo
campos_entrada = ['acousticness', 'danceability', 'duration_ms', 'instrumentalness',
                  'liveness', 'loudness', 'mode', 'speechiness', 'tempo']

# Escolher o campo alvo (popularidade)
campo_alvo = 'popularity'

# Pergunta interativa para escolher os campos de entrada
print("Escolha os campos de entrada digitando os números correspondentes (1 a 9):")
for i, campo in enumerate(campos_entrada, start=1):
    print(f"{i}. {campo}")

# Ler a entrada do usuário e converter para os índices dos campos
indices_escolhidos = input("Digite os números dos campos desejados (ex: 536): ")
indices_escolhidos = [int(idx) - 1 for idx in indices_escolhidos]

# Selecionar os campos escolhidos
campos_escolhidos = [campos_entrada[idx] for idx in indices_escolhidos]

# Separar dados de treino e teste
X = dataset[campos_escolhidos]
y = dataset[campo_alvo]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo MLP
modelo = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
modelo.fit(X_train, y_train)

# Fazer previsões
previsoes = modelo.predict(X_test)

# Avaliar o desempenho do modelo
erro_medio_quadratico = mean_squared_error(y_test, previsoes)
print(f'Erro Médio Quadrático: {erro_medio_quadratico}')

# Plotar gráficos para cada campo de entrada
plt.figure(figsize=(15, 12))
for i, campo in enumerate(campos_escolhidos, start=1):
    plt.subplot(3, 3, i)
    plt.scatter(X_test[campo], y_test, label='Real')
    plt.title(f'{campo.capitalize()} vs. Popularidade (Real)')
    plt.xlabel(campo.capitalize())
    plt.ylabel('Popularidade')
    plt.legend()

    plt.subplot(3, 3, i + 3)
    plt.scatter(X_test[campo], previsoes, label='Previsto', color='orange')
    plt.title(f'{campo.capitalize()} vs. Popularidade (Previsto)')
    plt.xlabel(campo.capitalize())
    plt.ylabel('Popularidade')
    plt.legend()

plt.tight_layout()
plt.show()
