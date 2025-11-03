# Encrypted NIDS
建議資料夾結構：

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