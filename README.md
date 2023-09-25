- 本篇希望透過 AWS Sagemaker 的全託管服務，輕鬆建立一套簡易的 MLFlow 流程，打造一個簡易系統，讓 ML 模型能夠做到即時服務。
- 本篇價值希望在於系統架構實作，例如**流量處理、模型部署、資料庫互動**等等，因此模型面並不會使用 SOTA 模型。
- 架構圖如下：
  

- 目標：
  1. Sagemaker 全託管流程跑一遍
  2. 自定義標記後資料＆自定義流程跑一遍
  3. 流量管控不同方案跑一遍