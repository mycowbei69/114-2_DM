# 第 14 週｜作業
- 修改課本第12章程式碼，運用AI製作修改程式記錄的MD檔
  - 修改了什麼
  - 該程式區塊的功能說明
- 放置於README.MD中

# W14-Exercise
## Error01
ValueError: 這個錯誤 ValueError: could not convert string to float: 'sales' 的意思是，當您嘗試計算 DataFrame 的相關性（df.corr()）時，Python 發現其中有非數值型的資料，例如 'sales' 這個欄位是字串（物件型別），而相關性只能對數值型資料進行計算。因為 'sales' 無法被轉換成浮點數，所以會拋出這個錯誤。

我已經在之前的步驟中，將程式碼修改為 sns.heatmap(df.select_dtypes(include=np.number).corr().round(2), annot=True, cmap='coolwarm');。這個修改的目的是使用 df.select_dtypes(include=np.number) 來篩選出 DataFrame 中所有的數值型欄位，這樣在計算相關性時，就不會再包含非數值型的 'sales' 或 'salary' 欄位，從而避免了這個錯誤。

您可以再次執行 TYIoq0XawiP6 這個儲存格，應該就能看到正確的熱力圖輸出，而不會再出現此錯誤了。
```
# sns.heatmap(df.corr().round(2), annot=True, cmap='coolwarm');
sns.heatmap(df.select_dtypes(include=np.number).corr().round(2), annot=True, cmap='coolwarm');
```
這個修改的目的是使用 df.select_dtypes(include=np.number) 來篩選出 DataFrame 中所有的數值型欄位，這樣在計算相關性時，就不會再包含非數值型的 'sales' 或 'salary' 欄位，從而避免了這個錯誤。
這行程式碼的目的是繪製一個顯示資料數值型欄位之間相關性關係的熱力圖。
- df.select_dtypes(include=np.number)：df 是您的 Pandas DataFrame。select_dtypes() 是一個 Pandas DataFrame 的方法，用於根據資料類型選擇或排除欄位。include=np.number 指定只選擇 DataFrame 中所有數值型（整數、浮點數等）的欄位。這樣做是為了確保只有可以用來計算相關性的數值資料才會被納入考量，避免了字串型欄位導致的錯誤。
- .corr()：這是 DataFrame 的另一個方法，用於計算所有選定數值型欄位之間的皮爾遜相關係數（Pearson correlation coefficient）。結果會是一個相關矩陣，顯示每對欄位之間的相關性。
- .round(2)：將相關矩陣中的數值四捨五入到小數點後兩位，使結果更易於閱讀。
- sns.heatmap(...)：sns 是 Seaborn 函式庫的別名，Seaborn 是一個基於 Matplotlib 的資料視覺化函式庫。heatmap() 是 Seaborn 中用於繪製熱力圖的函式。熱力圖是一種用顏色深淺來表示數值大小的圖表。									
- annot=True：這是 heatmap() 函式的一個參數，設定為 True 時，會在熱力圖的每個方格內顯示對應的相關係數數值，方便直接讀取。
- cmap='coolwarm'：這是 heatmap() 函式用來指定色彩映射的參數。'coolwarm' 是一種發散型色圖，通常用於表示從負值到正值的區間：冷色系（藍色）表示負相關，暖色系（紅色）表示正相關，中間顏色（白色/淺色）表示接近零的相關性。

## Error02
這個錯誤 ValueError: could not convert string to float: 'sales' 表示 corrwith() 函式嘗試對非數值型的 'sales' 欄位計算相關性，這是不允許的。就像之前的錯誤一樣，相關性計算需要數值資料。我將修改程式碼，使其只對數值型欄位計算與 'left' 欄位的相關性。
```
# df.drop('left', axis=1).corrwith(df['left']).round(2)
df.drop('left', axis=1).select_dtypes(include=np.number).corrwith(df['left']).round(2)
```
這些數值有助於我們理解哪些因素對員工離職的決策影響較大。
- satisfaction_level (滿意度): 與離職呈現顯著的負相關 (-0.39)，這表示員工滿意度越低，離職的可能性越高。
- last_evaluation (上次考評)、number_project (專案數量)、average_montly_hours (平均月工時) 與離職呈現微弱的正相關，表示這些因素稍微增加時，離職的可能性也略有上升。
- time_spend_company (在公司待的年數): 與離職呈現正相關 (0.14)，這可能暗示了在公司待太久也可能導致離職（例如職業倦怠或尋求新挑戰）。
- Work_accident (工作事故): 與離職呈現負相關 (-0.15)，表示曾有工作事故的員工離職的可能性較低。
- promotion_last_5years (過去五年是否晉升): 與離職呈現負相關 (-0.06)，表示最近有晉升的員工離職的可能性較低。
