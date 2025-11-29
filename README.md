# Encrypted NIDS
## 建議資料夾結構

```text
encrypted-NIDS/
├── bin/                       # 可執行檔或啟動腳本
├── config/                    # 設定檔 (yaml, json, toml)
├── data/                      # 原始封包或測試資料 (pcap 等)
├── docs/                      # 文件與設計說明
├── examples/                  # 範例配置與使用範例
├── keys/                      # 加密金鑰與憑證（注意不要提交敏感金鑰）
├── logs/                      # 運行日誌
├── scripts/                   # 部署、建構或輔助腳本
├── src/                       # 原始程式碼
│   ├── data_processing/       # 資料處理函式
│   ├── model/                 # 模型
│   └── utils/                 # 共用工具函式
├── tests/                     # 單元測試與整合測試
├── pyproject.toml | setup.py  # 專案建構設定（視語言而定）
├── requirements.txt           # 相依套件（若使用 Python）
├── LICENSE
└── README.md
```

## 命名規則
```
<用途>_<資料集>(_<模型>)(_<方式>)(_<二/多分類>)(_<版本>).<副檔名>
```
### 用途
 - dataProcessing: 資料處理
 - testing: 測試
 - training: 訓練

### 資料集
 - TON: The TON_IoT Datasets
 - USTC: USTC-TFC2016
 - CIC23: the CIC IoT 2023
 - CIC18: CSE CIC IDS 2018

### 模型
 - ViT
 - Swin

### 方式
 - Raw: 內容特徵
 - Statistical: 統計特徵
 - Both: 內容及統計特徵

### 二/多分類
 - Bin: 二分類
 - Muti: 多分類

## venv (可選)
### 建立
#### windows
`python -m venv <project>`
#### macOS/Linux
`python -m venv <project>`
### 啟動
#### windows
`<project>\Scripts\activate`
#### macOS/Linux
`source <project>/bin/activate`
