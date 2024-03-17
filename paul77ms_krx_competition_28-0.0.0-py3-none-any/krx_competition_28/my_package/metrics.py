import numpy as np
from sklearn.metrics import mean_squared_error

def MAPE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    실제 값과 예측 값 사이의 평균 절대 백분율 오류(MAPE)를 계산

    :param np.ndarray y_test: 실제 값
    :param np.ndarray y_pred: 예측 값
    :return: 평균 절대 백분율 오류(MAPE) 값
    :rtype: float
    """
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def MAE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    실제 값과 예측 값 사이의 평균 절대 오차(MAE)를 계산

    :param np.ndarray y_test: 실제 값
    :param np.ndarray y_pred: 예측 값
    :return: 평균 절대 오차(MAE) 값
    :rtype: float
    """
    return np.mean(np.abs(y_test - y_pred))

def RMSE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    실제 값과 예측 값 사이의 평균 제곱근 오차(RMSE)를 계산

    :param np.ndarray y_test: 실제 값
    :param np.ndarray y_pred: 예측 값
    :return: 평균 제곱근 오차(RMSE) 값
    :rtype: float
    """
    MSE = mean_squared_error(y_test, y_pred)
    return np.sqrt(MSE)
