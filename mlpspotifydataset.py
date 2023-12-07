from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from spotipy_fetch import get_spotify_features  # Importa a função do módulo spotipy_fetch.py

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

# Obter a URL da faixa do usuário
track_url = input("Digite a URL da faixa no Spotify: ")

# Obter as características da faixa usando a função do módulo spotipy_fetch.py
track_features = get_spotify_features(track_url)
print(track_features)

# Adicionar as características da faixa ao conjunto de teste
X_test_track = pd.DataFrame([track_features], columns=campos_escolhidos)

# Separar dados de treino e teste
X = dataset[campos_escolhidos]
y = dataset[campo_alvo]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Normalizar os dados de treino e teste
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalizar os dados da faixa do usuário
X_test_track_scaled = scaler.transform(X_test_track)

# Criar e treinar o modelo MLP com os dados normalizados
modelo = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=500, batch_size=500)
modelo.fit(X_train_scaled, y_train)

# Fazer previsões com dados de teste normalizados
previsoes = modelo.predict(X_test_scaled)

# Adicionar a previsão para a faixa do usuário
previsao_track = modelo.predict(X_test_track_scaled)

# Avaliar o desempenho do modelo
erro_medio_quadratico = round(mean_squared_error(y_test, previsoes), 2)
erro_medio_absoluto = round(mean_absolute_error(y_test, previsoes), 2)
print(f'Erro Médio Quadrático: {erro_medio_quadratico}')
print(f'Erro Médio Absoluto: {erro_medio_absoluto}')

# Plotar gráficos para cada campo de entrada
plt.figure(figsize=(15, 12))
for i, campo in enumerate(campos_escolhidos, start=1):
    plt.subplot(3, 3, i)
    plt.scatter(X_test_scaled[:, i-1], y_test, label='Real')
    plt.scatter(X_test_track_scaled[:, i-1], previsao_track, color='red', label='Faixa escolhida')  # Destacar em vermelho
    plt.title(f'{campo.capitalize()} vs. Popularidade (Real)')
    plt.xlabel(campo.capitalize())
    plt.ylabel('Popularidade')
    plt.legend()

    plt.subplot(3, 3, i + 3)
    plt.scatter(X_test_scaled[:, i-1], previsoes, label='Previsto', color='orange')
    plt.scatter(X_test_track_scaled[:, i-1], previsao_track, color='red', label='Faixa escolhida')  # Destacar em vermelho
    plt.title(f'{campo.capitalize()} vs. Popularidade (Previsto)')
    plt.xlabel(campo.capitalize())
    plt.ylabel('Popularidade')
    plt.legend()

plt.tight_layout()
plt.show()
