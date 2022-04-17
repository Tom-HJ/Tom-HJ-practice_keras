# 이번 프로젝트는 인공지능 구현 6단계를 따릅니다.
# 1단계: 분류 ANN 구현용 패키지 가져오기
# 2단계: 분류 ANN에 필요한 매개변수 설정
# 3단계: 분류 ANN 모델 구현
# 4단계: 학습과 성능 평가용 데이터 가져오기
# 5단계: 분류 ANN 학습 및 검증
# 6단계: 분류 ANN 학습 결과 분석

# 1단계: 분류 ANN 구현용 패키지 가져오기
from keras import layers, models

def ANN_models_func(Nin, Nh, Nout)
    # 입력 계층을 layers.Input() 함수로 지정합니다. 원소를 Nin개 가지는 입력신호 벡터는 입력 노드에 따른 계층의 shape를 (Nin,)로 지정합니다.
    x = layers.Input(shape=(Nin,))

    # 노드가 Nh개인 은닉 계층은 layers.Dense(Nh)로 지정합니다. x를 입력노드로 받아들이고 싶으면 뒤에 (x)를 추가합니다. 
    h = layers.Activation('relu')(layers.Dense(Nh)(x))
    
    # 출력 계층은 다음과 같이 지정합니다. 출력 노드수는 Nout으로 지정합니다. 이때 출력 노드에 입력되는 정보는 은닉노드의 출력값입니다.
    # 분류의 경우는 출력 노드의 활성화 함수로 소프트맥스 연산을 수행합니다.
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))
    
    model = models.Model(x, y)
    
    # 컴파일은 다음과 같이 구현합니다.
    # loss는 손실함수를 지정하는 인자입니다. 케라스가 제공하는 손실함수 외에도 직접 새로운 손실함수를 지정할 수 있습니다.
    # optimizer는 최적화 함수를 지정합니다.
    # metrics는 학습이나 예측이 진행될 때 성능 검증을 위해 손실 뿐 아니라 정확도도 측정하라는 의미입니다.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model