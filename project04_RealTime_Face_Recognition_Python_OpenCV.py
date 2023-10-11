## Step1 열굴검출 및 사진 수집하기

import cv2
import os

# 이미지 저장 폴더 생성 함수
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 이미지 저장 폴더 생성
image_directory = "training_images"
create_directory_if_not_exists(image_directory)

# 얼굴 검출을 위한 Haar cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 카메라 연결
cap = cv2.VideoCapture(0)

count = 0
while count < 100:
    # 카메라로부터 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    # 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        # 얼굴 부분만 잘라내기
        face = gray[y:y+h, x:x+w]
       
        
        # 이미지 저장
        cv2.imwrite(os.path.join(image_directory, "face{}.jpg".format(count)), face)
        count += 1

    # 화면에 결과 보여주기
    cv2.imshow('Collecting Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업 완료 후 자원 해제
cap.release()
cv2.destroyAllWindows()

## Step2 사진학습시키기
import cv2
import os
import numpy as np

# 학습 이미지 경로
image_directory = "training_images"

# 이미지와 레이블을 저장할 리스트
images = []
labels = []

# 이미지 파일을 순회하면서 읽어들이기
for i in range(100):
    image_path = os.path.join(image_directory, "face{}.jpg".format(i))
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:  # 이미지 파일이 정상적으로 읽혔는지 확인
        images.append(img)
        labels.append(0)  # 이 예제에서는 한 사람의 이미지만 사용하므로 레이블을 0으로 통일

# numpy 배열로 변환
images_np = [np.asarray(img, dtype=np.uint8) for img in images]
labels_np = np.asarray(labels, dtype=np.int32)

# LBPH 얼굴 인식기 생성 및 학습
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(images_np, labels_np)

# 학습 모델 저장
face_recognizer.save("face_model.yml")

print("학습완료!!!")


## Step3 얼굴 인식 예측하기
import cv2

# 학습된 모델 불러오기
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")

# 얼굴 검출을 위한 Haar cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 카메라 연결하여 실시간 영상 캡처 시작
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # 화면의 너비 설정
cap.set(4, 1080)  # 화면의 높이 설정

# 임의로 설정한 임계값 (이 값을 테스트를 통해 조절해야 할 수 있습니다.)
THRESHOLD = 60

while True:
    # 카메라로부터 현재 프레임 가져오기
    ret, frame = cap.read()
    if not ret:
        break

    # 흑백 이미지로 변환하여 얼굴 검출 성능 향상
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 검출된 얼굴 부분만 잘라내기
        face = gray[y:y+h, x:x+w]
        

        # 잘라낸 얼굴을 모델에 입력하여 인식 결과 얻기
        label, confidence = face_recognizer.predict(face)
        print(label)
        print(confidence)
        
        # 인식 결과가 학습된 인물이고, confidence 값이 임계값 이하라면
        if  confidence <= THRESHOLD:
            # if label == 0 and confidence <= THRESHOLD:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Access Granted", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {100 - confidence:.2f}%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Access Denied", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 현재 프레임에 인식 결과 표시하여 화면에 보여주기
    cv2.imshow('Face Recognition', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 화면 창 자원 해제
cap.release()
cv2.destroyAllWindows()
