requires: `python >= 3.9`

## 1. Quick Start
### Installation
```shell
$ pip install falcon_mark
```

### Run
```shell
$ init_falcon_mark
```

### View app in browser

Local URL: http://localhost:8501

## 2. How to run test server

### Install requirements
``` shell
pip install -e '.[test]'
```

### Run test server
```
$ python -m falcon_mark
```

OR:

```shell
$ init_falcon_mark
```

## 3. How to check coding styles

```shell
$ ruff check falcon_mark
```

Add `--fix` if you want to fix the coding styles automatically
```shell
$ ruff check falcon_mark --fix
```
