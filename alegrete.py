import numpy as np


def compute_mse(b, w, data):
    """
    Calcula o erro quadratico medio
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """
    # inicialização de variáveis
    squared_error_sum = 0

    input_values = data[:, 0]           # valores de x conhecidos
    goal_values = data[:, 1]            # valores de y conhecidos

    # cria um loop que calcula o erro para cada valor da amostra
    for input_value, goal_value in zip(input_values , goal_values):
        squared_error = compute_squared_error(b, w, input_value, goal_value)        # erro para um valor de x conhecido com (predição - valor obtido)^2
        squared_error_sum += squared_error                                          # soma dos erros

    data_values_count = len(data)                                       # quantidade de valores de x e y conhecidos
    mean_squared_error = squared_error_sum / data_values_count          # média do erro quadrado = soma dos erros / quantidade de valores conhecidos
    return mean_squared_error

def compute_squared_error(b, w, input_value, goal_value):
    prediction = w * input_value + b                    # predição = w * x + b -> regressão usando equação de primeiro grau
    squared_error = (prediction - goal_value) ** 2      # erro usando MSE
    return squared_error

def step_gradient(b, w, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de b e w.
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de b e w, respectivamente
    """
    
    # inicialização de variáveis
    bias_correction_factor = 0                  # fator de correção do bias
    weight_correction_factor = 0                # fator de correção do peso

    input_values = data[:, 0]                   # valores de x conhecidos
    goal_values = data[:, 1]                    # valores de y conhecidos
    number_of_values = len(data)                # quantidade de valores de x e y conhecidos
    derivative_factor = 2 / number_of_values    # valor extraído da derivada na equação de descida de gradiente

    # faz um loop que calcular o fator de correção do bias e do peso
    for input_value, goal_value in zip(input_values , goal_values):
        prediction = w * input_value + b                                                # predição = w * x + b -> regressão usando equação de primeiro grau
        absolute_error = prediction - goal_value                                        # erro absoluto = predição - valor obtido
        weight_correction_factor += derivative_factor * absolute_error * input_value    # fator de correção do peso = (2/N) * erro * x
        bias_correction_factor += derivative_factor * absolute_error                    # fator de correção do bias = (2/N) * erro
        
    corrected_weight = w - alpha * weight_correction_factor                             # atualiza o peso com um fator de passo alfa
    corrected_bias = b - alpha * bias_correction_factor                                 # atualiza o bias com um fator de passo alfa

    return corrected_bias, corrected_weight

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
