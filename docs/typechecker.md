# 型別驗證

## 開發中
vscode 可以使用微軟的 Pylance 及 My Type Checker，進行初步檢查，再來可使用以下 my 進行型別檢查。

### 安裝
```
pip install -r requirements-stubs.txt
pip install -r requirements-checktools.txt
```

## 使用
### mypy
```
mypy <path>
```
例如
```
mypy ./src/**/*.py
```


## git push
建議使用 pre-commit 先進行檢查
1. 安裝 pre-commit
```
pip install -r requirements-checktools.txt
```
2. 安裝設定檔至 pre-commit
```
pre-commit install
```