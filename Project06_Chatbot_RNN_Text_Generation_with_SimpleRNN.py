
# import and read file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

filepath = './robots.txt'  # Your file
corpus = open(filepath, 'r', errors='ignore')
raw_data = corpus.read()
print(raw_data)

# tokenizer
t = Tokenizer()
t.fit_on_texts([raw_data])
vocab_size = len(t.word_index) + 1
print(f'단어 집합의 크기: {vocab_size}')

print(t.word_index)  # 각  단어와  단어에  부여된  정수  인덱스  출력- 빈도수 순으로 출력

# sequence
sequences = list()
for line in raw_data.split('\n'):  # \n을 기준으로 문장 토큰화
    print(t.texts_to_sequences([line]))
    # text_to_sequences 의 0번째 위치에 문장의 순서가 들어있음
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
print(f'학습에 사용한 샘플 개수:{len(sequences)}')

print(sequences)  # 전체  샘플을  출력

max_len = max(len(l) for l in sequences)  # 모든  샘플에서  길이가  가장  긴  샘플의  길이  출력
print('샘플의  최대  길이  : {}'.format(max_len))

# padding
# 전체 샘플의 길이를 184로 패딩
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences)

# train & test data
sequences = np.array(sequences)
X = sequences[:, :-1]  # 학습 데이터
y = sequences[:, -1]  # 정답(Label) 데이터 print(X)
print(X.shape, y.shape)

# modeling

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len-1))  # 여기서 100은 임베딩 차원
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)  # epochs 튜닝 필요

# 문장 생성 함수


def sentence_generation(model, t, current_word, n):
    init_word = current_word  # 입력된 초기 단어를 저장합니다. 출력할 때 사용됩니다.
    sentence = ''  # 생성될 문장을 저장할 변수입니다.

    for _ in range(n):  # n번 반복합니다.
        # 현재 단어를 정수 인코딩합니다.
        encoded = t.texts_to_sequences([current_word])[0]
        # 문장의 길이를 맞추기 위해 패딩을 적용합니다. maxlen은 문장의 최대 길이입니다.
        encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre')
        # 모델을 사용하여 다음 단어를 예측합니다.
        result = np.argmax(model.predict(encoded), axis=-1)

        # 예측 결과를 단어로 변환합니다.
        for word, index in t.word_index.items():
            if index == result:
                break
        # 현재 단어에 예측 단어를 추가합니다.
        current_word = current_word + ' ' + word
        # 문장에 예측 단어를 추가합니다.
        sentence = sentence + ' ' + word

    sentence = init_word + sentence  # 초기 단어와 생성된 문장을 결합합니다.
    return sentence


# 함수를 테스트합니다.
print(sentence_generation(model, t, 'party', 10))
