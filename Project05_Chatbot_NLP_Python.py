

# ### 챗봇 만들기
# 1. 라이브러리 가져오기
# 2. 데이터 가져오기
# 3. 데이터 전처리하기
#    1) 소문자로 변환
#    2) 단어 토큰화하기 - word_tokenize()사용 or sent_tokenize()
#    3) 표제어 추출함수 정의하기 - nltk.stem.WordNetLemmatizer() 사용
#    4) 정규화 함수 정의하기 - 구두점 제거하고 표제어 추출하기
# 4. 응답함수 정의하기
#    1) 질문 문장을 기존 토큰에 추가하기
#    2) 토큰에 대해 tfidf 생성하기
#    3) 질문 문장과 기존 토큰과의 코사인 유사도 구하기
#    4) 코사인 유사도 값이 가장 큰 값 구하기
#    5) 코사인 유사도 값이 가장 큰 문장 + 유사도값 출력하기
# 5. 인삿말 대응 함수 정의하기 - greeting()
# 6. 끝인삿말 대응 함수 정의하기 - goodbye()
# 7. 챗봇 사용하기
#     1) 안내문 띄우기
#     2) 질문 입력받기
#     3) 질문 소문자 변경하기
#     4) 인삿말 - 인사말대응, 끝인삿말-끝인삿말대응, 질문-챗봇응대

# 필요한 라이브러리와 모듈을 먼저 가져옵니다.
import nltk
import numpy as np
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize

# 예시 데이터 (raw_data) 불러오기
# 실제 코드에서는 파일에서 불러와야 함
filepath = './robots.txt'
corpus = open(filepath, 'r', errors='ignore')
raw_data = corpus.read().lower()
sent_tokens = nltk.sent_tokenize(raw_data.lower())
# print(raw_data)
# print(sent_tokens)

# 표제어 추출 함수 정의
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# 정규화 함수 정의
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# 응답 함수 정의


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        return "I am sorry! I don't understand you"
    else:
        return sent_tokens[idx]


# 인삿말 및 끝인삿말 데이터
GREETING_INPUTS = ["hello", "안녕", "hi", "greetings",
                   "sup", "what's up", "hey", "hey there"]
GREETING_RESPONSES = ["hi", "hey", "안녕하세요", "hi there",
                      "hello", "I am glad! You are talking to me"]
GOODBYE_INPUTS = ["잘가", "bye", "goodbye", "see you", "later", "farewell"]
GOODBYE_RESPONSES = ["잘가", "Goodbye!", "See you later!",
                     "Farewell!", "Bye! Come back again soon."]

# 인삿말 및 끝인삿말 처리 함수


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def goodbye(sentence):
    for word in sentence.split():
        if word.lower() in GOODBYE_INPUTS:
            return random.choice(GOODBYE_RESPONSES)

# 메인 응답 함수


def main_response(user_input):
    if user_input.lower() in GREETING_INPUTS:
        return greeting(user_input)
    elif user_input.lower() in GOODBYE_INPUTS:
        return goodbye(user_input)
    else:
        return response(user_input)

# 안내문 출력


def show_guide():
    print("안녕하세요! 저는 당신의 챗봇입니다.")
    print("어떤 질문이든 물어보세요. 'bye'를 입력하면 종료됩니다.")


# 메인 실행 부분
if __name__ == "__main__":
    show_guide()
    while True:
        user_input = input("You: ").lower()
        if user_input.lower() in GOODBYE_INPUTS:
            print("Chatbot: 즐거웠어요. 안녕히가세요.")
            break
        else:
            print("Chatbot:", main_response(user_input))
