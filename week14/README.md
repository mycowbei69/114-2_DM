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
