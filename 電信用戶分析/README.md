# 電信用戶分析 (Telco Customer Churn)

## 一、專案背景

Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs  

## 二、資料集說明

- 資料來源：[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
- 資料內容： 7043 customers and 21 features  
- 資料前處理：缺失值處理、特徵標準化

## 三、技術與方法

- 使用 Python  套件  
- 資料分割： 測試集30%，訓練集70%
- 模型：KNN,SVC,隨機森林,邏輯迴歸,決策樹,Ado Boost,Voting Classifier
- 評估指標：混淆矩陣 (Confusion Matrix)、ROC 曲線

## 四、主要步驟

1. 讀取資料並了解資料型態  
2. 缺失值處理及資料標準化  
3. 資料視覺化  
4. 建立模型並訓練 

## 五、成果展示

- 資料分析(EDA)  
- 模型評估指標  
- 分析顯示:綜合 Recall 和 F1-score結果,邏輯迴歸方法較符合實際情況,但隨機森林模型的準確率較高。

## 六、參考
https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/notebook
