#공부시간으로 점수 예측하기

# 필요한 라이브러리 임포트
import numpy as np                          # 넘파이 임포트
import matplotlib.pyplot as plt             # 맷플롯립 임포트
import pickle
from tensorflow.keras import Sequential     # Sequential 클래스 임포트
from tensorflow.keras.layers import Dense   # Dense 클래스 임포트

# 데이터 준비하기
X=np.array([[12],[24],[27],[36],[48],[72],[96],[100]])       # 8행 1열의 2차원 배열을 입력 X에 할당
Y=np.array([[50],[70],[65],[75],[80],[90],[95],[100]]) # 8행 1열의 2차원 배열을 타깃 Y에 할당
print(X.ndim, ", X의 차원")
print(X.shape, ", X의 행렬 크기")

# 데이터 선형성 확인
plt.scatter(X, Y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 데이터 표준화
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X-X_mean)/X_std
print(X_mean, "X_mean")
print(X_std, "X_std")


ScoreML = Sequential()
ScoreML.add(Dense(1, input_shape=(1,)))

ScoreML.compile(loss='MSE', optimizer='SGD')
ScoreML.fit(X_scaled,Y,epochs=200)
value = ScoreML.predict(np.array([X_scaled[0],[(30-X_mean)/X_std],X_scaled[2]]))
print(value)