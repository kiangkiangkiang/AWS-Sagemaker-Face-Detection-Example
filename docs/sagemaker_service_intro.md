# Sagemaker 功能一覽

使用到的功能：
1. Notebook Instance
2. Ground Truth (用來標記資料)

## 資料（準備，前處理，標記，儲存）

### 資料準備

直接把準備好的資料放到 S3，等等標記時指向 S3 即可。

可以參考以下的官方範例：
<details><summary>官方範例</summary>
建立一個 Notebook Instance（或其他可以 access 到 aws 的環境，包含local terminal、local python file 等等，只要可以 access 都可以）

以 Sagemaker Notebook Instance 為例，建立 notebook instance 後，開啟一個 .ipynb file，執行以下程式，正常執行後即把資料 download 到 S3 了。
```python
import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()

!aws s3 sync s3://sagemaker-sample-files/datasets/image/caltech-101/inference/ s3://{bucket}/ground-truth-demo/images/

print('Copy and paste the below link into a web browser to confirm the ten images were successfully uploaded to your bucket:')
print(f'https://s3.console.aws.amazon.com/s3/buckets/{bucket}/ground-truth-demo/images/')

print('\nWhen prompted by Sagemaker to enter the S3 location for input datasets, you can paste in the below S3 URL')

print(f's3://{bucket}/ground-truth-demo/images/')

print('\nWhen prompted by Sagemaker to Specify a new location, you can paste in the below S3 URL')

print(f's3://{bucket}/ground-truth-demo/labeled-data/')
```
</details>

或手動把資料放入 S3：
1. 我們上網隨便搜尋 5 張包含人臉的照片，以存於[這邊](../data/)。
2. 把這 5 張包含人臉的照片放入指定的 S3（這裡不多做介紹，自己放進自己帳號的 S3 bucket，只要記得放在哪即可）

### 資料標記

#### Ground Truth -> Labeling jobs -> Create labeling jobs

選擇預先建立好的 S3 位置，任務類型選擇 bounding box。

**任務描述類型請詳盡清楚，後續 AWS 將會派遣人力資源幫你標記。**

預設是 5 個人力幫忙標記資料，計價方式假設是以此例子來做說明：
- 資料筆數: 5 份
- 審查費用區間: 5 筆資料 < 50000，因此費用為 0.08 USD
- 任務費用: 0.036 USD/每份資料
  
因此本次任務的費用為：任務費用＋審查費用（必須）
- 任務費用 = 0.036 * 5 (5 張圖片) * 5 (5 個標籤人員) = 0.9
- 審查費用 = 0.08 * 5 (5 張圖片，若超過 50000 張圖片，則多的圖片以 0.04 計價) = 0.4
- 總共 = 0.9 + 0.4 = 1.3 （換算成台幣約為40元）

計價表：https://aws.amazon.com/tw/sagemaker/data-labeling/pricing/

標記完後選擇「輸出資料及位置」，即可看到標記完資料儲存的 S3 位置，點選：manifests/output/output.manifest，此為 output 的 JSON Lines 格式的輸出資料。

--- 

## 模型訓練

### 演算法

將資料準備好後，我們即可開始訓練模型，Sagemaker 可選擇的訓練模型方式是：
1. 使用內建演算法（基本任務都有，基本上只要是來自 Ground Truth 標記完的資料都能無痛串接，最快最方便）
2. 自定義演算法（較為客製化的訓練方式，例如串接 LoRA 等等模塊）
3. 自己準備 Container（按照 Sagemaker 準備好後，放到 ECR）
4. 其他內建演算法

在 Sagemaker 主控台左側，選擇「培訓」->「訓練任務」->「建立訓練任務」

這裡使用 Sagemaker 內建的演算法：Vision - Object Detection (MxNet)




---

## 模型部署

---

## 模型推論

---

## 討論

1. 標記任務：這部分需要依靠 Sagemaker 所提供的人力、標記軟體等等，後續可研究：「如何在自己的標記環境中，自行進行標記作業，將標記結果串接回來，進行後續的模型訓練」，如此便能提供更客製化的標記作業與流程，並降低成本。
2. 模型訓練：我們所使用的是 Sagemaker 內建演算法，可以研究如何使用自訂義演算法做訓練。
## 總結


---
## 參考

1. 資料標記：https://aws.amazon.com/tw/getting-started/hands-on/machine-learning-tutorial-label-training-data/
2. 使用 Ground Truth 標記資料訓練：https://aws.amazon.com/tw/blogs/machine-learning/easily-train-models-using-datasets-labeled-by-amazon-sagemaker-ground-truth/
3. Sagemaker 演算法：https://docs.aws.amazon.com/zh_tw/sagemaker/latest/dg/algorithms-choose.html