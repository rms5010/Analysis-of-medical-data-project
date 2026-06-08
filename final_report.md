# AI 기반 수요 예측 및 물류 비용 최적화 프로젝트 최종 보고서

---

**학번:** 20242404  
**이름:** 유명석

---

## 목차

1. [프로젝트 소개](#1-프로젝트-소개)
2. [주제 관련 배경](#2-주제-관련-배경)
3. [데이터셋 소개](#3-데이터셋-소개)
4. [전처리 과정](#4-전처리-과정)
5. [모델 구조](#5-모델-구조)
6. [레퍼런스 및 개선점](#6-레퍼런스-및-개선점)
7. [프로젝트 결과](#7-프로젝트-결과)
8. [추후 발전 방향](#8-추후-발전-방향)

---

## 1. 프로젝트 소개

### 1.1 주제 선정 배경

이커머스 시장의 급성장과 함께 수요 예측의 정확성이 물류 운영 효율에 미치는 영향이 점점 중요해지고 있다. 기업은 수요를 과소 예측하면 품절로 인한 매출 손실과 고객 이탈이 발생하고, 과대 예측하면 재고 보관 비용이 증가하는 딜레마에 직면한다.

기존 연구 및 산업 현장에서는 수요 예측 모델의 성능을 주로 SMAPE, RMSE 등 통계적 지표로 평가한다. 그러나 이러한 지표가 낮다고 해서 실제 운영 비용도 반드시 최소화된다는 보장은 없다. 예측 오차의 **방향성(과대 vs 과소 예측)** 과 **비용 함수의 비대칭성** 이 존재하기 때문이다.

본 프로젝트는 다음의 핵심 연구 질문을 중심으로 설계되었다.

> **"예측 정확도(SMAPE)가 높을수록 실제 운영 비용도 반드시 감소하는가?"**

### 1.2 프로젝트 목표

- 브라질 실제 이커머스 데이터를 활용한 제품 카테고리별 수요 예측 모델 구축
- 전통 통계 모델(ARIMA), 머신러닝(XGBoost), 딥러닝(LSTM) 세 모델의 성능 비교
- 예측 결과를 재고 운영 비용(보관 비용 + 품절 비용)과 연결하는 비용 시뮬레이션 수행
- 예측 정확도와 운영 비용 최소화 사이의 관계 분석 및 인사이트 도출

### 1.3 프로젝트 구조

```
project/
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── predictions.csv          ← 모델링 결과 저장
│   └── smape_summary.csv        ← SMAPE 요약 저장
├── src/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_smape_evaluation.ipynb
│   ├── 05_cost_simulation.ipynb
│   └── 06_final_comparison.ipynb
└── output/
    └── final_dashboard.png
```

---

## 2. 주제 관련 배경

### 2.1 수요 예측의 중요성

수요 예측(Demand Forecasting)은 공급망 관리의 핵심 문제로, 과거 수요 데이터를 바탕으로 미래 수요를 추정하여 재고 수준, 생산 계획, 물류 배치를 결정하는 데 활용된다. 예측 오차가 클수록 다음 두 가지 비용이 발생한다.

- **재고 보관 비용(Holding Cost):** 예측이 실제보다 높아 재고가 남을 경우 발생하는 창고 보관, 자본 기회비용 등
- **품절 비용(Stockout Cost):** 예측이 실제보다 낮아 재고가 부족할 경우 발생하는 매출 손실, 고객 이탈 비용 등

일반적으로 품절 비용이 보관 비용보다 크기 때문에(s > h), 수요 예측은 약간의 과대 예측 방향으로 설계되는 경향이 있다.

### 2.2 뉴스벤더 모형(Newsvendor Model)

본 프로젝트의 발주 전략은 운영관리(Operations Management)의 고전 이론인 **뉴스벤더 모형**을 기반으로 한다. 수요가 확률적으로 분포할 때, 기대 비용을 최소화하는 최적 발주 분위수는 다음과 같다.

$$q^* = \frac{s}{s + h}$$

여기서 s는 품절 비용, h는 보관 비용이다. 수요가 정규분포를 따른다고 가정하면 최적 발주량은 다음과 같이 결정된다.

$$Q^* = \hat{\mu} + z^* \cdot \sigma, \quad z^* = \Phi^{-1}(q^*)$$

$\hat{\mu}$는 예측 수요, $\sigma$는 수요 표준편차, $\Phi^{-1}$은 표준정규분포의 역함수이다.

### 2.3 SMAPE와 비용의 관계

SMAPE는 예측 오차의 **크기**만을 측정하며 **방향성(과대/과소)**을 구분하지 않는다.

$$\text{SMAPE} = \frac{100}{n} \sum_{t=1}^{n} \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|) / 2}$$

반면 비용 함수는 비대칭적이다. 동일한 SMAPE를 가진 두 예측이라도, 어느 방향으로 틀렸느냐에 따라 발생하는 비용이 달라진다. 이것이 "SMAPE 1위 모델 ≠ 비용 최소 모델"이 될 수 있는 이론적 근거이다.

---

## 3. 데이터셋 소개

### 3.1 데이터 출처

**Brazilian E-Commerce Public Dataset by Olist**  
출처: Kaggle ([링크](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce))

브라질 최대 이커머스 플랫폼 Olist에서 실제 거래된 주문 데이터로, 2016년부터 2018년까지의 약 10만 건 이상의 주문 정보를 포함한다.

### 3.2 사용 파일 및 구조

| 파일명 | 주요 컬럼 | 설명 |
|--------|-----------|------|
| `olist_orders_dataset.csv` | order_id, order_purchase_timestamp | 주문 기본 정보 |
| `olist_order_items_dataset.csv` | order_id, product_id, freight_value | 주문 상품 및 배송비 |
| `olist_products_dataset.csv` | product_id, product_category_name | 상품 카테고리 |
| `olist_customers_dataset.csv` | customer_id, customer_state | 고객 정보 |
| `olist_order_payments_dataset.csv` | order_id, payment_value | 결제 정보 |

### 3.3 분석 범위

- **전체 데이터 기간:** 2016-09 ~ 2018-08
- **분석 적용 기간:** 2017-01-01 ~ 2018-08-31 (2016년 데이터는 수집 초기 노이즈로 제외)
- **학습(Train) 기간:** 2017-01-01 ~ 2018-06-30 (약 546일)
- **테스트(Test) 기간:** 2018-07-01 ~ 2018-08-31 (약 61일)
- **분석 카테고리:** 주문량 기준 상위 10개 카테고리

### 3.4 상위 10개 카테고리 (수요 기준)

주문량 합계 기준 상위 10개 제품 카테고리를 선정하여 분석하였다. 카테고리 선정 기준을 재고 수가 아닌 실제 주문량(demand)으로 설정함으로써 분석의 현실성을 높였다.

---

## 4. 전처리 과정

### 4.1 날짜 데이터 처리

`order_purchase_timestamp` 컬럼을 datetime 형식으로 변환하고, 이후 reindex 및 lag feature 계산 시 타입 오류를 방지하기 위해 `.dt.normalize()`를 사용하여 날짜 단위로 변환하였다.

```python
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_date'] = orders['order_purchase_timestamp'].dt.normalize()
```

`.dt.date` 대신 `.dt.normalize()`를 사용한 이유는 `datetime64` 타입을 유지해 이후 `reindex`, `resample`, lag feature 계산에서 발생하는 타입 불일치 오류를 방지하기 위함이다.

### 4.2 분석 기간 필터링

```python
ANALYSIS_START = '2017-01-01'
ANALYSIS_END   = '2018-08-31'

orders = orders[
    (orders['order_date'] >= ANALYSIS_START) &
    (orders['order_date'] <= ANALYSIS_END)
]
```

### 4.3 테이블 조인 및 결측치 처리

`orders`, `order_items`, `products` 세 테이블을 조인하여 주문별 카테고리 정보를 결합하였다. `product_category_name`이 결측인 행은 카테고리 분류가 불가능하므로 제거하였다.

```python
merged = pd.merge(orders[['order_id', 'order_date']],
                  order_items[['order_id', 'product_id', 'freight_value']],
                  on='order_id', how='inner')
merged = pd.merge(merged,
                  products[['product_id', 'product_category_name']],
                  on='product_id', how='left')
merged = merged.dropna(subset=['product_category_name'])
```

### 4.4 일별 수요 집계 및 날짜 갭 채우기

카테고리별 일별 주문 건수를 수요(demand)로 정의하고, 주문이 없는 날(수요 = 0)이 누락되지 않도록 전체 날짜 범위로 reindex하여 0으로 채웠다. 이 처리가 없으면 lag/rolling feature가 엉뚱한 날짜 간 계산되는 문제가 발생한다.

```python
full_date_range = pd.date_range(start=ANALYSIS_START, end=ANALYSIS_END, freq='D')

for cat in TOP_CATEGORY_LIST:
    cat_df = (demand_df[demand_df['product_category_name'] == cat]
              .set_index('order_date')[['demand']]
              .reindex(full_date_range, fill_value=0))
```

### 4.5 Feature Engineering

ARIMA는 원본 시계열을 그대로 사용하며, XGBoost와 LSTM을 위해 아래의 피처를 생성하였다. **카테고리 경계를 넘지 않도록 카테고리별로 독립적으로 계산**하였다.

| 피처 | 설명 |
|------|------|
| `lag_1` | 전일 수요 |
| `lag_7` | 7일 전 수요 (동일 요일 효과) |
| `lag_14` | 14일 전 수요 (2주 전 패턴) |
| `rolling_mean_7` | 7일 이동평균 (단기 추세) |
| `rolling_std_7` | 7일 이동표준편차 (단기 변동성) |
| `rolling_mean_14` | 14일 이동평균 (중기 추세) |
| `weekday` | 요일 (0=월 ~ 6=일) |
| `is_weekend` | 주말 여부 |
| `month` | 월 |
| `day_of_year` | 연중 일수 (계절성) |

rolling 피처 계산 시 `shift(1)`을 먼저 적용하여 현재 시점의 수요가 피처에 포함되는 **data leakage를 방지**하였다.

```python
df['rolling_mean_7'] = df['demand'].shift(1).rolling(window=7, min_periods=1).mean()
```

### 4.6 Train/Test Split

시계열 데이터의 특성상 무작위 분할(random split)은 미래 데이터가 학습에 포함되는 data leakage를 유발한다. **반드시 시간 순서 기반으로 분할**하였다.

```python
train_raw = demand_filled[demand_filled['order_date'] <= TRAIN_END]  # ~ 2018-06-30
test_raw  = demand_filled[demand_filled['order_date'] >= TEST_START] # 2018-07-01 ~
```

### 4.7 LSTM 시퀀스 생성

LSTM 입력을 위해 window_size=14의 슬라이딩 윈도우로 `(samples, timesteps, features)` 형태의 3D 배열을 생성하였다. 스케일링은 train 데이터 기준으로만 `fit`하고 test에는 `transform`만 적용하여 leakage를 방지하였다. test 시퀀스 생성 시 train 마지막 14일을 앞에 붙여 경계 연속성을 확보하였다.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
cat_train_scaled = scaler.fit_transform(cat_train.reshape(-1, 1)).flatten()
cat_test_scaled  = scaler.transform(cat_test.reshape(-1, 1)).flatten()

combined_scaled = np.concatenate([cat_train_scaled[-WINDOW_SIZE:], cat_test_scaled])
```

---

## 5. 모델 구조

### 5.1 ARIMA (AutoRegressive Integrated Moving Average)

전통적인 시계열 예측 모델로, 자기회귀(AR), 차분(I), 이동평균(MA) 세 요소를 결합한다.

**파라미터 설정:**
- ADF(Augmented Dickey-Fuller) 검정으로 정상성 여부를 판단하여 차분 차수 d를 결정하였다. p-value < 0.05이면 정상 시계열(d=0), 아니면 d=1을 적용하였다.
- p=1, q=1로 고정하여 카테고리별 독립 모델을 학습하였다.

**예측 방식:** Rolling (Walk-forward) Forecast  
매 스텝마다 실제값을 history에 추가하여 예측의 현실성을 높였다. 음수 예측값은 0으로 클리핑하였다.

```python
for t in range(len(test)):
    model  = ARIMA(history, order=order)
    fitted = model.fit()
    yhat   = fitted.forecast(steps=1)[0]
    predictions.append(max(0.0, yhat))
    history.append(test[t])
```

### 5.2 XGBoost

Gradient Boosting 기반의 앙상블 머신러닝 모델로, 비선형 관계와 복잡한 피처 상호작용 포착에 강점이 있다.

**설계 특징:**
- 10개 카테고리를 하나의 통합 모델로 학습하여 카테고리 간 패턴을 공유하고 샘플 부족 문제를 완화하였다.
- `product_category_name`을 레이블 인코딩하여 피처로 추가하였다.

**주요 하이퍼파라미터:**

| 파라미터 | 값 |
|----------|-----|
| n_estimators | 500 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 3 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |

### 5.3 LSTM (Long Short-Term Memory)

순환 신경망(RNN)의 일종으로, 장기 의존성(long-term dependency)을 학습하는 데 적합한 딥러닝 모델이다.

**모델 구조:**
```
LSTM(64, return_sequences=True)
    → Dropout(0.2)
    → LSTM(32, return_sequences=False)
    → Dropout(0.2)
    → Dense(1)
```

**학습 설정:**

| 설정 | 값 |
|------|----|
| Window Size | 14일 |
| Epochs | 최대 100 (EarlyStopping 적용) |
| Batch Size | 16 |
| Validation Split | 0.1 |
| EarlyStopping Patience | 10 |
| Optimizer | Adam |
| Loss | MSE |

카테고리별 수요 스케일이 달라 카테고리별 독립 모델을 학습하였다.

### 5.4 평가 지표 — SMAPE

수요가 0인 날이 존재하여 일반 MAPE의 분모가 0이 되는 문제를 해결하기 위해 SMAPE를 사용하였다. 실제·예측 모두 0인 경우 오차는 0으로 처리하였다.

$$\text{SMAPE} = \frac{100}{n} \sum_{t=1}^{n} \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|) / 2}$$

### 5.5 비용 시뮬레이션

**비용 구조:**

$$\text{TotalCost} = \text{HoldingCost} + \text{StockoutCost}$$

$$\text{HoldingCost}  = \max(\text{Order} - \text{Demand},\ 0) \times h$$

$$\text{StockoutCost} = \max(\text{Demand} - \text{Order},\ 0) \times s$$

**비용 파라미터 설정:**
- 재고 보관 비용 h: 카테고리별 평균 배송비 × H_SCALE(0.1). 배송비가 높은 카테고리는 부피·무게가 크다는 가정 하에 보관 비용도 높게 설정하였다.
- 품절 비용 s: h × S_RATIO(3.0). 기회비용과 고객 이탈을 반영하여 보관 비용의 3배로 설정하였다.

**발주 전략 — Newsvendor 최적화:**
수요 불확실성을 반영하여 뉴스벤더 모형 기반의 최적 발주 분위수를 적용하였다.

$$Q^* = \hat{\mu} + \Phi^{-1}\left(\frac{s}{s+h}\right) \cdot \sigma$$

**비교 기준선:**
- **Perfect:** 예측이 완벽할 때 newsvendor 전략을 적용한 비용 (예측 오차 제거 시 기대 비용)
- **Naive:** 전일 수요를 그대로 발주량으로 사용하는 단순 baseline

---

## 6. 레퍼런스 및 개선점

### 6.1 주요 참고 문헌 및 기술

- Box, G.E.P. & Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control.* — ARIMA 모델 이론
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD. — XGBoost 알고리즘
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation. — LSTM 아키텍처
- Newsvendor Model — 운영관리(Operations Management) 고전 이론, 최적 재고 수준 결정
- Makridakis, S. et al. (2018). *Statistical and Machine Learning forecasting methods: Concerns and ways forward.* PLOS ONE. — 예측 모델 비교 방법론

### 6.2 프로젝트 중 발견 및 개선한 사항

| 항목 | 기존 방식 | 개선 방식 | 개선 이유 |
|------|----------|----------|----------|
| 날짜 변환 | `.dt.date` | `.dt.normalize()` | datetime64 타입 유지, reindex 오류 방지 |
| 카테고리 선정 기준 | 재고 수 기준 | 실제 주문량(demand) 기준 | 분석 현실성 향상 |
| Rolling 피처 계산 | `.rolling().mean()` | `.shift(1).rolling().mean()` | Data leakage 방지 |
| Rolling std (비용 시뮬레이션) | test 실제값 기반 | shift(1) 적용으로 lookahead 제거 | 현실적 시뮬레이션 |
| Naive baseline | 미포함 | 포함 | 비교 기준선 확보 |
| 비용 구조 | SMAPE 평가 미연결 | 뉴스벤더 기반 비용 시뮬레이션 연결 | 비즈니스 임팩트 분석 |

### 6.3 한계점

- **데이터 희소성:** 분석 기간이 약 20개월로, 카테고리별로 나누면 LSTM 학습에 충분하지 않을 수 있다. LSTM 성능이 상대적으로 낮은 것도 이와 관련이 있을 가능성이 있다.
- **ARIMA 파라미터 고정:** p=1, q=1로 고정하여 카테고리별 최적 파라미터를 탐색하지 못하였다. auto_arima를 적용하면 성능이 개선될 수 있다.
- **Baseline 비교 부재:** Naive SMAPE와의 비교가 이루어지지 않아, 세 모델이 단순 baseline 대비 우수한지 명확히 확인하지 못하였다.
- **통계적 유의성 검정 미수행:** ARIMA(47.27%) vs XGBoost(47.85%)의 차이가 통계적으로 유의한지 Diebold-Mariano test 등으로 검정하지 않았다.
- **단변량 예측:** 카테고리 간 수요 상관관계(cross-category spillover)를 고려하지 않았다.

---

## 7. 프로젝트 결과

### 7.1 예측 성능 비교 (SMAPE)

| 모델 | 평균 SMAPE | 비고 |
|------|-----------|------|
| **ARIMA** | **47.27%** | 전체 평균 최저 |
| XGBoost | 47.85% | ARIMA와 근소한 차이 |
| LSTM | 51.38% | 데이터 볼륨 부족으로 상대적 열세 |

전체 평균으로는 ARIMA가 가장 우수하지만, 카테고리별로 보면 일부에서 XGBoost 또는 LSTM이 더 낮은 SMAPE를 기록하였다. 단일 모델이 모든 카테고리에서 최적이 아님을 확인하였다.

### 7.2 운영 비용 비교 (TotalCost)

| 모델 | 총 운영 비용 (BRL) | 보관 비용 | 품절 비용 |
|------|------------------|----------|----------|
| **ARIMA** | **10,437** | - | - |
| XGBoost | 10,618 | - | - |
| LSTM | 12,080 | - | - |
| Perfect | (이론적 하한) | - | - |
| Naive | (baseline) | - | - |

Newsvendor 최적 발주 전략 적용 기준. 비용 기준으로도 ARIMA가 전체 합산에서 가장 낮았다.

### 7.3 핵심 발견 — "SMAPE 1위 ≠ 비용 1위"

SMAPE와 TotalCost의 상관관계 분석 결과, **약한 양의 상관관계**가 확인되었다. 이는 예측 정확도가 높을수록 대체로 비용이 낮아지는 경향은 있으나, 반드시 성립하는 것은 아님을 의미한다.

카테고리별 분석에서 **SMAPE 1위 모델과 비용 1위 모델이 불일치하는 "역전" 카테고리**가 일부 존재하였다.

**역전이 발생하는 이론적 원인:**
- SMAPE는 예측 오차의 크기만 측정하며 방향성(과대/과소)을 구분하지 않는다.
- 비용 함수는 비대칭적이다 (s > h, 품절이 더 비쌈).
- 동일한 SMAPE라도 어느 방향으로 틀렸느냐에 따라 발생 비용이 달라진다.

### 7.4 h·s 민감도 분석 결과

h_scale ∈ {0.05, 0.10, 0.20}, s_ratio ∈ {2.0, 3.0, 5.0}의 9가지 시나리오에서 비용 시뮬레이션을 반복하였다.

**비용 시뮬레이션의 rolling_std 개선:** 초기 구현에서 test 기간 실제값 기반의 rolling_std를 사용하여 lookahead bias 가능성이 있었다. `shift(1)` 적용으로 수정하였으나, 결과에 유의미한 차이가 없었다. 이는 단기 수요의 변동성이 안정적인 패턴을 보이기 때문으로, **본 분석의 결론이 해당 구현 방식에 민감하지 않음을 확인한 결과**이기도 하다.

### 7.5 종합 결론

> **"예측 정확도(SMAPE)가 높다고 해서 반드시 운영 비용이 최소화되지는 않는다."**

- 전체 평균 기준으로는 ARIMA가 SMAPE와 TotalCost 모두에서 가장 안정적인 성능을 보였다.
- 그러나 일부 카테고리에서는 SMAPE 1위 모델이 비용 1위가 아닌 역전 현상이 발생하였다.
- h·s 파라미터 민감도 분석에서도 ARIMA가 대부분의 시나리오에서 비용 기준 우위를 유지하였다.
- 비용 최소화를 목표로 한다면 모델 선택 기준을 SMAPE 단독이 아닌 TotalCost 직접 평가로 전환하는 것이 합리적이다.

---

## 8. 추후 발전 방향

### 8.1 모델 고도화

- **auto_arima 도입:** pmdarima 라이브러리를 활용하여 카테고리별 최적 (p,d,q) 파라미터를 자동 탐색. 현재 고정 파라미터(1,d,1)의 한계 보완.
- **Prophet 모델 추가:** Facebook Prophet은 계절성과 휴일 효과를 자동으로 처리하여 이커머스 수요 예측에 적합할 수 있다.
- **Transformer 기반 모델:** Temporal Fusion Transformer(TFT) 등 최신 시계열 딥러닝 모델과의 비교.

### 8.2 비용 시뮬레이션 고도화

- **카테고리별 비대칭 비용 파라미터:** 현재 모든 카테고리에 동일한 S_RATIO를 적용하였으나, 카테고리별 고객 민감도에 따라 차별화 가능.
- **동적 안전재고:** 계절성, 이벤트(블랙프라이데이 등)에 따라 안전재고 수준을 동적으로 조정하는 모형 적용.
- **리드타임 반영:** 발주에서 입고까지의 리드타임을 고려한 실제 재고 운영 시뮬레이션.

### 8.3 분석 확장

- **Naive SMAPE 비교 추가:** 단순 baseline 대비 각 모델의 예측 성능 우위를 명확히 검증.
- **Diebold-Mariano Test 적용:** 모델 간 예측 성능 차이의 통계적 유의성 검정.
- **카테고리 간 교차 수요 분석:** 관련 카테고리 간 수요 상관관계를 활용한 다변량 예측 모형 적용.
- **손실 함수 직접 최적화:** 비용 함수를 손실 함수로 직접 사용하여 비용 최소화에 특화된 모델 학습 (Cost-sensitive Learning).

---

*본 보고서는 2024학년도 AI/데이터분석 프로젝트 최종 결과물입니다.*
