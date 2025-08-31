# 糖尿病預測模型 (Diabetes Prediction Model)

## 一、專案背景

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.  
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.  
Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.  

## 二、資料集說明

- 資料來源：[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)  
- 資料內容：共有768筆記錄，包含8個輸入特徵（如血糖、BMI、年齡等）及一個二元標籤（是否患糖尿病）  
- 資料前處理：缺失值處理、特徵標準化

## 三、技術與方法

- 使用 Python 及 GridSearchCV 套件  
- 資料分割：訓練集66%、測試集34% 
- 模型：KNN模型  
- 評估指標：準確率 (Accuracy)、混淆矩陣 (Confusion Matrix)、ROC 曲線及 AUC

## 四、主要步驟

1. 讀取資料並初步探索  
2. 缺失值處理及資料標準化  
3. 建立KNN模型並訓練  
4. 預測測試集並計算評估指標  
5. 模型調整與驗證

## 五、成果展示

- 模型準確率達到約 77.2%（實際數字填入）  
- AUC 指標為 0.819（實際數字填入）  
- 分析顯示 BMI 與血糖濃度為重要特徵

## 六、參考
https://www.kaggle.com/code/raja75/starter-pima-indians-diabetes-database-e1f30fb2-d
