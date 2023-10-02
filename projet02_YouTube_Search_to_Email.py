import smtplib
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime

# 현재 날짜와 시간을 가져옵니다.
current_time = datetime.now().strftime('%d/%m/%Y %H:%M')

# Initialize YouTube API
api_key = "api key"  # Put your YouTube API key here
youtube = build('youtube', 'v3', developerKey=api_key)

# Fetch YouTube data based on keyword and sorting condition ('date' or 'viewCount')


def fetch_youtube_data(keyword, order):
    request = youtube.search().list(
        q=keyword,
        part='snippet',
        maxResults=5,  # Number of results to retrieve
        order=order
    )
    response = request.execute()
    return response['items']

# Process the data to construct the email body


def process_data(items):
    email_body = ""
    for item in items:
        title = item['snippet']['title']
        description = item['snippet']['description']
        video_id = item['id']['videoId']
        video_link = f"https://www.youtube.com/watch?v={video_id}"
        # Link embedded in title
        email_body += f"<a href='{video_link}'>{title}</a>,<br>"
        email_body += f"Description: {description}<br><br>"
    return email_body

# Send an email


def send_email(subject, body):
    try:
        from_email = "your@gmail.com"  # Replace with your Gmail address
        to_email = "your@gmail.com"  # Replace with the recipient's email address
        password = "your pw"  # Replace with your Gmail password

        msg = MIMEMultipart()
        msg['From'] = from_email  # Fixed variable name
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)  # Fixed variable name
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully!")
    except Exception as e:
        print(f"Email could not be sent. Error: {e}")


# Main function to tie it all together
def main():
    keywords = ['deep learning', 'tech news']
    email_body_total = ""
    for keyword in keywords:
        for order in ['date', 'viewCount']:
            items = fetch_youtube_data(keyword, order)
            email_body = process_data(items)
            condition = "Latest" if order == 'date' else "Most Viewed"
            email_body_total += f"{condition} YouTube Videos for [{keyword}]:<br>{email_body}<br>"

    send_email(f"{current_time}미니님! YouTube Updates입니다! {', '.join(keywords)} 관련 뉴스",
               email_body_total)  # Changed to include success or failure messages


if __name__ == '__main__':
    main()
