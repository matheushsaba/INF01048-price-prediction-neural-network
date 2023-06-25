import numpy as np


def compute_mse(b, w, data):
    """
    Calcula o erro quadratico medio
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """

    error_sum = 0                       # inicialização de uma variável que guardará a soma dos erros
    input_values = data[:, 0]           # valores de x conhecidos
    goal_values = data[:, 1]            # valores de y conhecidos

    # cria um loop que calcula o erro para cada valor da amostra
    for input_value, goal_value in zip(input_values , goal_values):
        error = compute_error(b, w, input_value, goal_value)        # erro para um valor de x conhecido com (predição - valor obtido)^2
        error_sum += error                                          # soma dos erros

    data_values_count = len(data)                               # quantidade de valores de x e y conhecidos
    mean_squared_error = error_sum / data_values_count          # média do erro quadrado = soma dos erros / quantidade de valores conhecidos
    return mean_squared_error

def compute_error(b, w, x, goal_value):
    prediction = w * x + b                   # predição = w * x + b -> regressão usando equação de primeiro grau
    error = (prediction - goal_value) ** 2   # erro usando MSE
    return error

def step_gradient(b, w, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de b e w.
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de b e w, respectivamente
    """

    input_values = data[:, 0]           # valores de x conhecidos
    goal_values = data[:, 1]            # valores de y conhecidos

    # faz um loop que atualiza o bias e o peso com cada valor da amostra
    for input_value, goal_value in zip(input_values , goal_values):
        error = compute_error(b, w, input_value, goal_value)        # erro para um valor de x conhecido com (predição - valor obtido)^2
        w = w - alpha * error * input_value                         # atualiza o peso com um fator de passo alfa
        b = b - alpha * error                                       # atualiza o bias com um fator de passo alfa

    return b, w


def fit(data, b, w, alpha, num_iterations):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de b e w.
    Ao final, retorna duas listas, uma com os b e outra com os w
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os b e outra com os w obtidos ao longo da execução
    """
    raise NotImplementedError  # substituir pelo seu codigo
