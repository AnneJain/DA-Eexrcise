#安裝 mlxtend套件
!pip install mlxtend
#匯入函數
from mlxtend.plotting import plot_decision_regions  ##視覺化分類模型
import numpy as np ##矩陣、向量
import pandas as pd ##資料處理、表格
import matplotlib.pyplot as plt ##基本繪圖套件
import seaborn as sns ##matplotlib進階套件
sns.set()  ##套用 seaborn 的預設繪圖風格

##讀取 Google 雲端硬碟中的檔案
from google.colab import drive

##匯入Python警告控制模組，並設定忽略警告訊息
import warnings
warnings.filterwarnings('ignore')

##直接顯示在 cell 下面
%matplotlib inline

#Loading the dataset
diabetes_data = pd.read_csv('/content/drive/MyDrive/備審資料/Data/Pima_Indians_Diabetes_Database.csv')

#Print the first 5 rows of the dataframe.
diabetes_data.head()

##檢查資料表結構
diabetes_data.info(verbose=True)

##匯出統計摘要，了解數值型欄位的分布
diabetes_data.describe().T

##處理資料中的異常值,複製一份新資料(diabetes_data_copy)進行修改,不會異動到rawdata(diabetes_data)
diabetes_data_copy = diabetes_data.copy(deep = True)

##缺失值標記:欄位中0值換成NaN
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

## showing the count of Nans 
print(diabetes_data_copy.isnull().sum())

##了解資料分佈以利估算nan值
p = diabetes_data.hist(figsize = (20,20))

##缺失值處理:以平均數,中位數替換成nan的值
#平均數
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
#中位數
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

##替換後圖形分佈
p = diabetes_data_copy.hist(figsize = (20,20))

##顯示資料欄數/列數
diabetes_data.shape

##計算每種型態的欄位數
dtype_counts = diabetes_data.dtypes.value_counts()
##畫 bar chart
sns.barplot(x=dtype_counts.values, y=dtype_counts.index)
plt.xlabel("Count of each data type")
plt.ylabel("Data types")
plt.show()

##missingno套件:缺失值視覺化分析,每個欄位缺失值的長條圖
import missingno as msno
p=msno.bar(diabetes_data)

##計算每個類別數量
counts = diabetes_data.Outcome.value_counts()
total = counts.sum()
##顏色設定
colors = ["#0392cf", "#7bc043"]
##畫柱狀圖
ax = counts.plot(kind="bar", color=colors)
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.title("Distribution of Diabetes Outcome")

##加百分比標籤
for i, count in enumerate(counts):
    percentage = count / total * 100
    ax.text(i, count + 5, f"{percentage:.1f}%", ha='center')

plt.show()

##數值欄位的散佈圖矩陣(scatter matrix),觀察特徵之間的關係
from pandas.plotting import scatter_matrix

#設定顏色: 0 → 藍色（無糖尿病）1 → 綠色（有糖尿病）
colors = diabetes_data['Outcome'].map({0: '#0392cf', 1: '#7bc043'})

#畫散佈圖矩陣
scatter_matrix(diabetes_data, figsize=(25, 25), diagonal='hist', alpha=0.6, marker='o', color=colors)

plt.suptitle("Scatter Matrix of Diabetes Dataset with Outcome Colors", fontsize=20)
plt.show()

##用Seaborn的pairplot取代傳統scatter_matrix，且用Outcome類別作為顏色標記
p = sns.pairplot(diabetes_data_copy, hue='Outcome', palette={0: '#0392cf', 1: '#7bc043'},
                 diag_kind='kde', plot_kws={'alpha':0.6, 's':50})

##相關係數熱圖:rawdata
plt.figure(figsize=(12,10))
#Pearson correlation
p = sns.heatmap(diabetes_data.corr(), annot=True, cmap='RdYlGn', linewidths=0.5)
plt.title("Correlation Heatmap of Diabetes Dataset", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

##相關係數熱圖:完成缺失值處理
plt.figure(figsize=(12,10))
#Pearson correlation
p = sns.heatmap(diabetes_data_copy.corr(), annot=True, cmap='RdYlGn', linewidths=0.5)
plt.title("Correlation Heatmap of Diabetes Dataset", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

##特徵標準化(Feature Scaling/Standardization):所有數值特徵做z-score標準化,產生新的DataFrame
#StandardScaler:把數值欄位轉換成平均值為0,標準差為1的標準化資料
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()

##取出目標欄位Outcome存成變數/y,y是一個Series，包含0（無糖尿病）和 1（有糖尿病）,作為模型訓練時的標籤（Label/Target）
y = diabetes_data_copy.Outcome

##訓練集/測試集切分(Train/Test Split)
#train_test_split 函數，將資料集分成訓練集和測試集
from sklearn.model_selection import train_test_split
##資料分群設定
#1. X:特徵矩陣（標準化資料）Y:目標欄位（Outcome）
#2. 1/3測試,2/3訓練
#3. 固定隨機種子，使切分結果可重現
#4. 按照y的類別比例切分資料:確保訓練集和測試集的糖尿病（1）與非糖尿病（0）比例一致
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

##使用KNN模型訓練:距離的分類方法，預測新樣本時會看最近k個鄰居的類別
from sklearn.neighbors import KNeighborsClassifier

test_scores = []#測試集準確率
train_scores = []#訓練集準確率

#KNN的鄰居數:1~14
for i in range(1,15):

    knn = KNeighborsClassifier(i)#建立 KNN 模型，鄰居數設定為 i
    knn.fit(X_train,y_train) #用訓練資料訓練KNN模型

    #knn.score():計算準確率,找出最佳k值,平衡訓練與測試的準確率
    train_scores.append(knn.score(X_train,y_train))#訓練集準確率
    test_scores.append(knn.score(X_test,y_test))#測試集準確率

##訓練集準確率最高的k值
max_train_score = max(train_scores) #訓練集最佳分數

#列表找出哪個索引i的準確率等於最高準確率
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
#轉成百分比表示,且將索引(從0開始)轉換成對應的k值
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

##測試集準確率最高的k值
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

##畫出k值對應的準確率曲線
plt.figure(figsize=(12,5))
p = sns.lineplot(x=range(1,15), y=train_scores, marker='*', label='Train Score')
p = sns.lineplot(x=range(1,15), y=test_scores, marker='o', label='Test Score')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs. Number of Neighbors')
plt.legend()#設定圖例
plt.show()#顯示圖表

##圖表結果顯顯示:當k=11,二個資料集準確率最接近,故最終模型使用k=11資料

#建立KNN分類器
knn = KNeighborsClassifier(11)#鄰居數k=11

#訓練資料訓練KNN模型並儲存
knn.fit(X_train,y_train)
#測試集準確率:在未見過的資料上正確預測的比例
knn.score(X_test,y_test)


##選擇最重要的兩個特徵(使用與Outcome相關性最大的兩個)
corr_matrix = diabetes_data_copy.corr()
top_features = corr_matrix['Outcome'].abs().sort_values(ascending=False).index[1:3]
print(f"Selected features for 2D decision boundary: {list(top_features)}")

X_2d = diabetes_data_copy[top_features].values

##分割2D特徵資料
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y.values, test_size=1/3, random_state=42, stratify=y
)

##建立 KNN 模型
knn_2d = KNeighborsClassifier(11)
knn_2d.fit(X_train_2d, y_train_2d)

#決策邊界
plt.figure(figsize=(10,6))
plot_decision_regions(X_train_2d, y_train_2d, clf=knn_2d, legend=1)
plt.scatter(X_test_2d[:,0], X_test_2d[:,1], c=y_test_2d, edgecolor='k', s=100, marker='X', label='Test Data')
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title("KNN Decision Boundary (Top 2 Features)")
plt.legend(loc='upper left')
plt.show()

#用sklearn的confusion_matrix計算KNN模型的預測結果
from sklearn.metrics import confusion_matrix


y_pred = knn.predict(X_test)#使用KNN模型進行預測
confusion_matrix(y_test,y_pred)#NumPy array格式的混淆矩陣
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)#有列名、欄名以及總和的表格

y_pred = knn.predict(X_test) #預測
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred) #計算混淆矩陣
#圖表
plt.figure(figsize=(6,5))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()

##KNN模型的精確度(precision)、召回率(recall)、F1-score及整體準確率
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
##數據說明:
#1. 類別不平衡：0 類別樣本數多（167 vs 89），模型對0類的表現比1類好。
#2. 召回率偏低：1 類召回率0.61(模型漏掉約39%的實際1類樣本)。
#精確度 vs 召回率：
#類別 0：精確度略低於召回率,偶爾誤把1類預測成 0。
#類別 1：精確度高於召回率,預測為1的結果較可靠，但漏掉許多實際1類。

##KNN模型的ROC曲線資料
from sklearn.metrics import roc_curve, roc_auc_score
##預測為正類的機率
y_pred_proba = knn.predict_proba(X_test)[:,1] #取得每個樣本屬於正類（1）的預測機率
#計算ROC曲線
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) #roc_curve:三個向量：fpr（假陽性率）、tpr（真正率）、thresholds（決策閾值）
#計算AUC
auc_score = roc_auc_score(y_test, y_pred_proba) #roc_auc_score:計算ROC曲線下的面積(AUC)，數值越接近1，模型越好
print(f"AUC Score: {auc_score:.3f}")

# 畫 ROC 曲線
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0,1], [0,1], color='red', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

##使用GridSearchCV調整KNN的n_neighbors超參數
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#設定要搜尋的超參數範圍 (n_neighbors 從 1 到 49)
param_grid = {'n_neighbors': np.arange(1, 50)}
#建立GridSearchCV，使用5折交叉驗證
knn_cv = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
#對整個資料集 X, y 進行調參
knn_cv.fit(X, y)
#顯示最佳分數與最佳參數
print("Best Score:" + str(knn_cv.best_score_)) 
print("Best Parameters: " + str(knn_cv.best_params_))
