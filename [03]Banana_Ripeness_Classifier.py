# Keeping feature folder
# 필요한 라이브러리를 임포트합니다.
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import os
import warnings

# 경고 메시지를 무시합니다.
warnings.filterwarnings('ignore')

# 폴더에서 이미지와 파일명을 로드하는 함수를 정의합니다.
def load_images_from_folder(folder):
    images = []
    filenames = []  # 파일 이름을 저장할 리스트
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)  # 파일 이름 저장
    return images, filenames  # 이미지와 파일 이름을 반환합니다.

# 각각의 폴더에서 이미지를 로드합니다.
black_bananas, _ = load_images_from_folder('overripeimg')
green_bananas, _ = load_images_from_folder('unripeimg')
yellow_bananas, _ = load_images_from_folder('fitripeimg')

# 평균 색상을 계산하는 함수를 정의합니다.
def averagecolor(image):
    return np.mean(image, axis=(0, 1))

# 학습 데이터와 레이블을 준비합니다.
trainX = []
trainY = []

# 레이블과 함께 이미지를 로드하고 특성을 추출합니다.
labels = ("black", "green", "yellow")
for bananas, label in zip((black_bananas, green_bananas, yellow_bananas), labels):
    for banana in bananas:
        trainX.append(averagecolor(banana))
        trainY.append(label)

# 레이블을 숫자로 변환합니다.
encoder = LabelEncoder()
encoded_trainY = encoder.fit_transform(trainY)

# SVM 모델을 학습합니다.
model_svm = svm.SVC(gamma="scale", decision_function_shape='ovr')
model_svm.fit(trainX, encoded_trainY)

# KNN 모델을 학습합니다.
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(trainX, encoded_trainY)

# 테스트 이미지와 파일명을 로드합니다.
test_images, test_filenames = load_images_from_folder('test_images')

# 예측을 수행합니다.
predictedY_svm = []
predictedY_knn = []

for img, filename in zip(test_images, test_filenames):  # 파일명도 함께 처리합니다.
    img_features = averagecolor(img)
    prediction_svm = model_svm.predict([img_features])[0]
    prediction_knn = knn_model.predict([img_features])[0]
    prediction_svm = encoder.inverse_transform([prediction_svm])[0]
    prediction_knn = encoder.inverse_transform([prediction_knn])[0]
    
    predictedY_svm.append(prediction_svm)
    predictedY_knn.append(prediction_knn)
    
    print(f"{filename}: SVM - {prediction_svm}, KNN - {prediction_knn}")

# 실제 테스트 레이블을 설정합니다. (수동으로 설정해야 함)
realtestY = ['black']*3 + ['green']*3 + ['yellow']*3

# 정확도 평가 함수를 정의합니다.
def evaluate_accuracy(predictedY):
    correct_predictions = np.sum(np.array(realtestY) == np.array(predictedY))
    total_predictions = len(realtestY)
    accuracy = (correct_predictions / total_predictions) * 100
    return f"Correct: {correct_predictions}. Wrong: {total_predictions - correct_predictions}. Correctly Classified: {accuracy:.2f}%"


# 분류 보고서를 출력합니다.
print("\nSVM Classification Report:")
print(classification_report(realtestY, predictedY_svm))

print("\nKNN Classification Report:")
print(classification_report(realtestY, predictedY_knn))

# 정확도 평가 출력
print("SVM Accuracy:")
print(evaluate_accuracy(predictedY_svm))

print("KNN Accuracy:")
print(evaluate_accuracy(predictedY_knn))



# X, Y train_test_split

from sklearn.model_selection import train_test_split  # 데이터를 학습용과 테스트용으로 분할하기 위한 라이브러리
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 라이브러리
from sklearn import svm  # SVM 라이브러리
from sklearn.neighbors import KNeighborsClassifier  # KNN 라이브러리
from sklearn.preprocessing import LabelEncoder  # 레이블을 숫자로 변환하기 위한 라이브러리
from sklearn.metrics import classification_report  # 분류 결과를 출력하기 위한 라이브러리
import cv2  # 이미지 처리를 위한 OpenCV 라이브러리
import numpy as np  # 수학적 연산을 위한 라이브러리
import os  # 파일과 디렉토리 작업을 위한 라이브러리

# 폴더에서 이미지와 파일명을 로드하는 함수를 정의합니다.
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images  # 이미지 리스트를 반환합니다.

# 평균 색상을 계산하는 함수를 정의합니다.
def averagecolor(image):
    return np.mean(image, axis=(0, 1))

# 모든 이미지와 레이블을 저장할 리스트를 초기화합니다.
all_images = []
all_labels = []

# 각 폴더와 해당 폴더의 레이블입니다.
folders = ['overripeimg', 'unripeimg', 'fitripeimg']
labels = ['black', 'green', 'yellow']

# 각 폴더에서 이미지를 로드하고 레이블을 부여합니다.
for folder, label in zip(folders, labels):
    images = load_images_from_folder(folder)
    all_images.extend(images)
    all_labels.extend([label] * len(images))

# 모든 이미지에서 특성을 추출합니다.
all_features = [averagecolor(img) for img in all_images]

# 레이블을 숫자로 변환합니다.
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(all_labels)

# 데이터를 학습용과 테스트용으로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(all_features, encoded_labels, test_size=0.2, random_state=42)

# SVM 모델을 만들고 학습합니다.
model_svm = svm.SVC(gamma="scale", decision_function_shape='ovr')
model_svm.fit(X_train, y_train)

# KNN 모델을 만들고 학습합니다.
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# 랜덤 포레스트 모델을 만들고 학습합니다.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측을 수행합니다.
predictedY_svm = model_svm.predict(X_test)
predictedY_knn = knn_model.predict(X_test)
predictedY_rf = rf_model.predict(X_test)

# 숫자 레이블을 다시 문자 레이블로 변환합니다.
predictedY_svm = encoder.inverse_transform(predictedY_svm)
predictedY_knn = encoder.inverse_transform(predictedY_knn)
predictedY_rf = encoder.inverse_transform(predictedY_rf)
y_test = encoder.inverse_transform(y_test)

# 분류 결과를 출력합니다.
print("\nSVM Classification Report:")
print(classification_report(y_test, predictedY_svm))

print("\nKNN Classification Report:")
print(classification_report(y_test, predictedY_knn))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, predictedY_rf))

# 정확도 평가 함수를 정의합니다.
def evaluate_accuracy(y_true, predictedY):
    correct_predictions = np.sum(np.array(y_true) == np.array(predictedY))
    total_predictions = len(y_true)
    accuracy = (correct_predictions / total_predictions) * 100
    return f"Correct: {correct_predictions}. Wrong: {total_predictions - correct_predictions}. Correctly Classified: {accuracy:.2f}%"

# 정확도를 출력합니다.
print("SVM Accuracy:")
print(evaluate_accuracy(y_test, predictedY_svm))

print("KNN Accuracy:")
print(evaluate_accuracy(y_test, predictedY_knn))

print("Random Forest Accuracy:")
print(evaluate_accuracy(y_test, predictedY_rf))
