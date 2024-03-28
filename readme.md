# Avaliação de Imagens usando Modelo de Classificação

Este repositório contém um script Python que utiliza um modelo de classificação de imagens treinado para avaliar imagens em uma determinada pasta. O modelo é carregado a partir de um arquivo HDF5 e é capaz de classificar imagens como "carro" ou "avião".

## Requisitos

Certifique-se de ter instalado os seguintes pacotes Python:

- TensorFlow
- Keras
- NumPy
- Matplotlib

Você pode instalar os pacotes usando o pip:

pip install tensorflow keras numpy matplotlib

## Utilização

1. **Preparação do Ambiente**: Baixe o repositório e certifique-se de que todas as imagens que você deseja avaliar estão presentes na pasta `img/`.

2. **Execução do Script**: Execute o script Python `avaliacao_imagens.py`. Este script irá carregar o modelo treinado a partir do arquivo HDF5, percorrer todas as imagens na pasta `img/`, fazer previsões para cada imagem e gerar uma imagem contendo cada imagem avaliada junto com o resultado da avaliação.

3. **Visualização dos Resultados**: Após a execução do script, você verá a imagem gerada contendo todas as imagens avaliadas juntamente com a classificação de cada uma delas.

## Observações

- Certifique-se de que o modelo HDF5 (`modelo.h5`) está presente no diretório raiz do projeto antes de executar o script.

Espero que este guia seja útil para você! Sinta-se à vontade para fazer qualquer pergunta ou fornecer feedback adicional.