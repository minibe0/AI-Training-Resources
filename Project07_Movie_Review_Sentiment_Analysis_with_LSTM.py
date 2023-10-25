import tensorflow as tf
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 데이터 로딩 및 전처리 ---
# Naver Sentiment Movie Corpus 다운로드
path_to_train_file = tf.keras.utils.get_file(
    'train.txt', 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt')
path_to_test_file = tf.keras.utils.get_file(
    'test.txt', 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt')

# 파일을 읽고 문장을 정제
with open(path_to_train_file, "rb") as f:
    train_text = f.read().decode(encoding='utf-8').split('\n')[1:]
train_text_X = [line.split('\t')[1]
                for line in train_text if len(line.split('\t')) > 1]
train_Y = np.array([int(line.split('\t')[-1])
                   for line in train_text if len(line.split('\t')) > 1]).reshape(-1, 1)

with open(path_to_test_file, "rb") as f:
    test_text = f.read().decode(encoding='utf-8').split('\n')[1:]
test_text_X = [line.split('\t')[1]
               for line in test_text if len(line.split('\t')) > 1]
test_Y = np.array([int(line.split('\t')[-1])
                  for line in test_text if len(line.split('\t')) > 1]).reshape(-1, 1)

# --- 텍스트 데이터 정제 함수 ---
# 정규 표현식을 이용한 텍스트 데이터 정제 (Cleaning)


def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower()


# 데이터 정제
train_text_X = [clean_str(sentence) for sentence in train_text_X]
test_text_X = [clean_str(sentence) for sentence in test_text_X]

# --- 텍스트 데이터 토큰화 ---
# 단어 토큰화 및 패딩
max_words = 20000  # 사용할 최대 단어 수 (빈도수가 높은 단어부터 사용)
max_length = 25  # 문장의 최대 길이

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_text_X)
train_X = pad_sequences(tokenizer.texts_to_sequences(
    train_text_X), maxlen=max_length, padding='post')
test_X = pad_sequences(tokenizer.texts_to_sequences(
    test_text_X), maxlen=max_length, padding='post')

# --- 모델 구성 ---
# LSTM을 사용한 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=max_words, output_dim=300, input_length=max_length),  # 단어 임베딩
    tf.keras.layers.LSTM(50),  # LSTM 레이어 (50개의 뉴런)
    tf.keras.layers.Dense(2, activation='softmax')  # 출력 레이어 (긍정/부정)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(train_X, train_Y, epochs=5, validation_split=0.2)

# --- 모델 평가 ---
# 테스트 데이터로 모델 평가
accr = model.evaluate(test_X, test_Y, verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(
    accr[0], accr[1]))

# --- 감성 분석 함수 ---
# 실시간 감성 분석을 위한 함수


def sentiment_predict(sentence):
    sentence = clean_str(sentence)
    tokenized_sentence = sentence.split(' ')
    truncated_sentence = tokenized_sentence[:25]
    sequence = tokenizer.texts_to_sequences([truncated_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=25, padding='post')
    prediction = model.predict(padded_sequence)
    positive_prob = prediction[0][1]
    if positive_prob > 0.5:
        print(f"{positive_prob * 100:.2f}% 확률로 긍정 리뷰입니다.")
    else:
        print(f"{(1 - positive_prob) * 100:.2f}% 확률로 부정 리뷰입니다.")


# 함수 테스트
sentiment_predict('이 영화 개꿀잼 ~')

print(sentiment_predict('이 영화 개꿀잼 ~'))
print(sentiment_predict('넘 재미없어 내내 졸았어요'))
print(sentiment_predict('돈이 아까워요 '))
print(sentiment_predict('이 영화 하품만 나와요~'))
print(sentiment_predict('이 영화 핵노잼 ㅠㅠ'))
print(sentiment_predict('이 영화 왜 만든거야'))
print(sentiment_predict('이 영화 꼭 보세요'))
print(sentiment_predict('안녕하세요'))
print(sentiment_predict('그저 그래요'))
print(sentiment_predict('좋아하는 사람들이 있을지는 모르겠지만 나는 그저그랬다'))
