# Encrypted NIDS
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

## pcapfix 安裝 (可選)
1. 下載[安裝檔](https://f00l.de/pcapfix/)
2. 解壓縮`tar zxvf pcapfix-<version>.tar.gz`
3. 進入資料夾`cd pcapfix-<version>.tar.gz`
4. 執行安裝
```
make
make install
```
5. 測試`pcapfix`

## 套件安裝
```
pip install -r requirements.txt
```
[pytorch](https://pytorch.org/get-started/locally/) (torch、torchvision、torchaudio) 需依照 GPU 安裝

