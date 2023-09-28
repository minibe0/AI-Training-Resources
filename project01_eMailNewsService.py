# 필요한 라이브러리를 불러옵니다.
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# 현재 날짜와 시간을 가져옵니다.
current_time = datetime.now().strftime('%d/%m/%Y %H:%M')

# Naver API 설정값을 변수에 저장합니다. (네이버 개발자센터에서 생성한 api)
client_id = "id"
client_secret = "secret"

# 이메일 설정값을 변수에 저장합니다. (사용할 메일계정과 비밀번호를 여기에 넣으세요, 샘플코드는 gmail로 설정함)
from_email = "example@gmail.com"
from_email_password = "pw"
to_email = "@gmail.com"

# 이메일을 보내는 함수를 정의합니다.
def send_email(email_subject, email_body):
    msg = MIMEMultipart()  # 이메일 객체 생성
    msg['From'] = from_email  # 보내는 사람 정보
    msg['To'] = to_email  # 받는 사람 정보
    msg['Subject'] = email_subject  # 이메일 제목 설정

    msg.attach(MIMEText(email_body, 'plain'))  # 이메일 본문 첨부

    server = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail SMTP 서버 설정
    server.starttls()  # TLS 보안 시작
    server.login(from_email, from_email_password)  # Gmail 로그인
    text = msg.as_string()  # 이메일 객체를 문자열로 변환
    server.sendmail(from_email, to_email, text)  # 이메일 전송
    server.quit()  # SMTP 서버 종료

# Naver 뉴스 검색 함수를 정의합니다.
def search_news(query):
    url = "https://openapi.naver.com/v1/search/news.json"  # Naver 뉴스 검색 API URL
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }  # API 헤더 설정
    params = {
        "query": query,
        "display": 5,  # 5개의 뉴스 결과를 가져옵니다.
    }  # 검색 파라미터 설정

    response = requests.get(url, headers=headers, params=params)  # API 요청
    news_data = response.json()  # 응답을 JSON 형태로 파싱

    return news_data  # 검색 결과 반환

# 메인 함수 시작
if __name__ == "__main__":
    try:
        keywords = ["경제","기술트랜드","인공지능"]  # 검색할 키워드 설정
        all_news_data = []  # 모든 뉴스 데이터를 저장할 리스트

        # 각 키워드에 대해 뉴스 검색을 수행합니다.
        for keyword in keywords:
            news_data = search_news(keyword)  # 뉴스 검색
            all_news_data.append((keyword, news_data))  # 키워드와 뉴스 데이터를 튜플로 저장

            # 이메일 제목과 본문 초기 설정
            email_subject = f"[{current_time} ]  {', '.join(keywords)} 관련 뉴스"
            email_body = f"  맞춤 뉴스! 네이버에서 퍼올렸어요!\n\n\n  '{', '.join(keywords)}':\n\n"

        # 각 키워드별 뉴스를 이메일 본문에 추가합니다.
        for keyword, news_data in all_news_data:
            email_body += f"--- {keyword} 관련 뉴스 ---\n\n"
            for item in news_data['items']:
                title = item['title']
                link = item['link']
                description = item['description']
                email_body += f"- {title}\n  {link}\n  {description}\n\n"

        # 이메일을 전송합니다.
        send_email(email_subject, email_body)
        print(f"Email sent successfully for keywords: {', '.join(keywords)}")

    except Exception as e:  # 예외 처리
        print(f"An error occurred: {str(e)}")



