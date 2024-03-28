# 데이터 분석 관련 라이브러리 불러오기
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# 데이터 시각화 관련 라이브러리 불러오기
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid') # matplotlib 스타일

# 그래프 출력
%matplotlib inline

# Scikit-Learn의 다양한 머신러닝 모듈 불러오기
from sklearn.linear_model import LogisticRegression # 선형회귀
from sklearn.svm import SVC, LinearSVC # 서포트벡터머신
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트
from sklearn.neighbors import KNeighborsClassifier # K-최근접이웃 알고리즘




# 데이터 가져오기
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

# 데이터 미리보기
train_df.head()




# 훈련데이터 정보 확인하기
train_df.info()
print('-'*20)

# 테스트데이터 정보 확인하기
test_df.info()




# 필요없는 컬럼 제거 (훈련데이터에서만)
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis=1)


# Pclass(좌석 등급)의 unique한 값 카운팅
train_df['Pclass'].value_counts()



# one-hot-encoding(pd.get_dummies 메서드)으로 범주형 데이터 인코딩
pclass_train_dummies = pd.get_dummies(train_df['Pclass'])
pclass_test_dummies = pd.get_dummies(test_df['Pclass'])

# Pclass의 원본을 없애고 범주형으로 개별로 데이터 변환
train_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)

train_df = train_df.join(pclass_train_dummies)
test_df = test_df.join(pclass_test_dummies)



# one-hot-encoding(pd.get_dummies 메서드)으로 범주형 데이터 인코딩
sex_train_dummies = pd.get_dummies(train_df['Sex'])
sex_test_dummies = pd.get_dummies(test_df['Sex'])

# Female/Male로 나눠서 지정
sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']

# Sex의 원본을 없애고 범주형으로 개별로 데이터 변환
train_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)

train_df = train_df.join(sex_train_dummies)
test_df = test_df.join(sex_test_dummies)




# 결측치(NaN) 보간을 위해 훈련데이터셋의 '평균값'으로 훈련 및 테스트데이터셋을 채워서 원본 변경
train_df["Age"].fillna(train_df["Age"].mean() , inplace=True)
test_df["Age"].fillna(train_df["Age"].mean() , inplace=True)



# 탑승료 컬럼 결측치 0으로 보간
test_df["Fare"].fillna(0, inplace=True)



# 객실 컬럼 결측치 버리기
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)




# 탑승항구 데이터 확인해보기
train_df['Embarked'].value_counts()




test_df['Embarked'].value_counts()



# S/C/Q 중 S가 대부분이므로 결측치는 S로 채워 보간한다.
train_df["Embarked"].fillna('S', inplace=True)
test_df["Embarked"].fillna('S', inplace=True)



# one-hot-encoding(pd.get_dummies 메서드)으로 범주형 데이터 인코딩
embarked_train_dummies = pd.get_dummies(train_df['Embarked'])
embarked_test_dummies = pd.get_dummies(test_df['Embarked'])

# S/C/Q로 나눠서 지정
embarked_train_dummies.columns = ['S', 'C', 'Q']
embarked_test_dummies.columns = ['S', 'C', 'Q']

# Embarked의 원본을 없애고 범주형으로 개별로 데이터 변환
train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)

train_df = train_df.join(embarked_train_dummies)
test_df = test_df.join(embarked_test_dummies)





# 탑승자정보/생존여부 <- 이 형태로 만들어주기 위해 데이터를 나눈다
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()


X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)



# 맨 위에서 불러운 머신러닝 알고리즘을 각각 적용해본다
# (1) 로지스틱 회귀
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

# score 메소드 : 분류 모델의 정확도를 리턴
logreg.score(X_train, Y_train)





# (2) 서포트 벡터 머신
svc = SVC()
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

# score 메소드 : 분류 모델의 정확도를 리턴
svc.score(X_train, Y_train)



# (3) 랜덤 포레스트
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

# score 메소드 : 분류 모델의 정확도를 리턴
random_forest.score(X_train, Y_train)



# (4) K-최근접이웃 알고리즘
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

# score 메소드 : 분류 모델의 정확도를 리턴
knn.score(X_train, Y_train)




# 모델 정확도 분석 결과 랜덤 포레스트가 가장 좋은 결과를 나타냄
# 해당 결과로 제출용 파일 만들기

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred    
    })

submission.to_csv('titanic.csv', index=False)
