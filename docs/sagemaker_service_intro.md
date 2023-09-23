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




### 資料前處理

### 資料儲存

--- 

## 模型建置


---

## 模型部署

---

## 模型推論

---

## 參考

1. https://aws.amazon.com/tw/getting-started/hands-on/machine-learning-tutorial-label-training-data/
