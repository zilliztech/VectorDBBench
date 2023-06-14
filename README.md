requires: `python >= 3.11`

## 1. Quick Start
### Installation
```shell
$ pip install vectordb-bench
```

### Run
```shell
$ init_bench
```

### View app in browser

Local URL: http://localhost:8501

## 2. How to run test server from source code

### Install requirements
``` shell
pip install -e '.[test]'
```

### Run test server
```
$ python -m vectordb_bench
```

OR:

```shell
$ init_bench
```

## 3. How to check coding styles

```shell
$ ruff check vectordb_bench
```

Add `--fix` if you want to fix the coding styles automatically
```shell
$ ruff check vectordb_bench --fix
```
