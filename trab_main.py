import os
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# caminho do data set
root = "dataSet/v20220930"

# inicia listas para as imagens e suas labels (rotulos)
X, y = [], []

# funcao para processar imagens
def careggar_imagem(caminho_imagem, tamanho_img=(64, 64)):
    try:
        imagem = Image.open(caminho_imagem).convert('L')  # tons de cinza
        imagem = imagem.resize(tamanho_img) # altera o tamanho da imagem
        return np.array(imagem).flatten() / 255.0  # normalizando a intensidade do pixel para o range de 0 - 1
    except Exception as e:
        print(f"Erro carregando imagem: {caminho_imagem}: {e}")
        return None

# carrega todas as imagens e suas labels
for pasta_classe in os.listdir(root):
    caminho_classe = os.path.join(root, pasta_classe)
    label = pasta_classe  # Nome da classe é o nome da pasta
    # Busca na subpasta (train_...) as imagens
    for subpasta in os.listdir(caminho_classe):
        caminho_subpasta = os.path.join(caminho_classe, subpasta)
        for imagem in os.listdir(caminho_subpasta): #pega todas as imagens da subpasta
            caminho_imagem = os.path.join(caminho_subpasta, imagem)
            imagem_array = careggar_imagem(caminho_imagem) # chama função para carregar e processar imagens
            if imagem_array is not None:
                #mapeia imagem X para label y
                X.append(imagem_array)
                y.append(label) 

if len(X) == 0:
    print("Erro, imagens não encontradas")
else:
    # converte as listas em arrays do numpy
    X = np.array(X)
    y = np.array(y)

    # Divide dados em treino e teste com otimizador adam
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', random_state=42, max_iter=25)
    #treina modelo
    mlp_clf.fit(X_train, y_train)

    # Eval do modelo, Presicao/recall/f1 macro para cada classe e acurancia
    y_pred = mlp_clf.predict(X_test)
    print("Resultados por classe:\n", classification_report(y_test, y_pred, zero_division=1))
    print("Matrix dos testes:\n", confusion_matrix(y_test, y_pred))

    # Para prever uma imagem qualquer
    def predict_custom_image(pasta_teste, classificador, tamanho_img=(64, 64)):
        previsoes = []
        #normaliza a imagem também
        for imagem in os.listdir(pasta_teste):
            img_array = careggar_imagem(os.path.join(pasta_teste, imagem), tamanho_img)
            if img_array is not None:
                #faz o modelo "prever"
                previsao = classificador.predict([img_array])
                previsoes.append(f"{previsao[0]} {'verdadeiro' if previsao[0] in ['I_u', 'i_l'] else 'falso'}")
        return previsoes

    # Caminho das imagens removidas do dataset para o modelo prever
    pasta_teste = "test_model"
    classe_prevista = predict_custom_image(pasta_teste, mlp_clf)
    print(f"A previsão retornou: {classe_prevista}")