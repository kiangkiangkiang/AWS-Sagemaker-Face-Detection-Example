# 如何連接 sagemaker studio 到 vscode 上

## 1. 建立 Vscode 環境

1. 點選 **System terminal**

![](./docs/sagemaker_home.png)

2. 在 Terminal 上，輸入以下指令：

```Shell
curl -LO https://github.com/aws-samples/amazon-sagemaker-codeserver/releases/download/v0.2.0/amazon-sagemaker-codeserver-0.2.0.tar.gz
tar -xvzf amazon-sagemaker-codeserver-0.1.5.tar.gz

cd amazon-sagemaker-codeserver/install-scripts/studio
 
chmod +x install-codeserver.sh
./install-codeserver.sh

```

3. 重整頁面，點選 **Code Server** 進入 vscode，之後就可以在 vscode 上 Coding 了！

*BTW，如果新增一個 file 要很久，建議重開 studio 或是 code-server，應該能夠解決。*


### 參考
1. 建立環境：https://aws.amazon.com/tw/blogs/machine-learning/host-code-server-on-amazon-sagemaker/


