EDA에서 얻은 인사이트를 바탕으로 결측치 처리를 우선적으로 진행

현재, 결측치가 존재하는 컬럼
- `users.csv`
    - age(27,833 : 약 41%)
- `books.csv`
    - language(67,227 : 약 45%)
    - category(68,857: 약 46%)
    - summary(67,227 : 약 45%)
    - img_path

> `summary, img_path`의 경우에는 각각 책의 요약 내용과 책의 커버 이미지이기 때문에 결측치를 채우는 것이 불가능하고 판단되어 그대로 사용

# 1. Age
- age 컬럼의 경우, baseline에서는 10년 단위로 age를 Binning한 뒤, 그 Mean값으로 대체
    - Age의 분포가 정규 분포와 비슷한 형태를 띄기 때문에위 방법도 가능하고 합리적임 (_EDA 참고_)
- 우리 팀에서는 Age의 확률분포에 맞게 Random Choice하는 방식으로 결측치를 채움
    - 위 방식이 원본 데이터의 특징을 가장 잘 유지할 수 있는 방법이라고 생각

# 1. Language 

0. books.csv의 book_title컬럼이 해당 row의 language값과 동일한 언어로 구성되어있다는 것을 확인
    - *e.g. Harry Potter -> English / 토지 -> Korean*
1. book_title의 값에서 20개의 언어 속성에 대한 feature를 추출
2. PCA(95%)를 통해 20의 속성 중 주요 속성을 추출
3. 추출된 주성분 feature를 활용해 Random Forest Classifier에 학습(약 99%의 정확도) 후 추론

[Language Classifier](https://github.com/NIckmin96/Book-Rating-Prediction/blob/main/preprocessing_bk/language_classifier.py)

# 2. Category 
Category의 경우에는 Language와 달리, 다른 Feature를 직접적으로 활용해서 결측치를 채울 수 없음

## IDEA
> 'Category도 다른 feature들을 직,간접적으로 활용해서 결측치를 채울 수 있지 않을까?'

1. `Category` 외의 다른 feature들을 numerical encoding(integer)
2. Embedding Table을 형성해 벡터화
3. 각 feature를 input으로, category 값을 output으로 하는 MLP Layer(3 Layers) 구성
4. Train(약 55% acc) & Inference

[Category Classifier](https://github.com/NIckmin96/Book-Rating-Prediction/blob/main/preprocessing_bk/cat_classifier.py)

# Conclusion

- 위의 3개의 feature에 대한 결측치를 채운 데이터를 Baseline 모델의 수정없이 그대로 사용하였을 때, Significant한 성능의 향상을 보임
