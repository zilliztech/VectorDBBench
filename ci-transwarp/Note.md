CI

---

### 准备数据集

```
aws s3 ls s3://assets.zilliz.com/benchmark/ --region us-west-2 --recursive --no-sign-request

aws s3 cp s3://assets.zilliz.com/benchmark/cohere_medium_1m cohere_medium_1m  --region us-west-2 --recursive --no-sign-request
```

### 构建

```shell
docker build \
    --network=host \
    -f ci-transwarp/Dockerfile \
    -t 172.16.1.99/hippo/vectordb_bench/builder \
    .
```

### 运行

```shell
git clone -b dev "http://gitlab+deploy-token-54:AJJ9dcXoYsHXKaHLdb2A@172.16.1.41/distributed-storage/vectordbbench.git"

# docker run这个在上一个clone出来的目录下跑
docker run \
    --network=host \
    -itd \
    -v $(pwd):/opt/transwarp/vectordb_bench \
    -v XXXX:/tmp/vectordb_bench/dataset \
    172.16.1.99/hippo/vectordb_bench/builder bash
```

XXXX这个目录是数据集的目录，目录结构大概如下（参考tw-node45节点/mnt/disk1/hippo/dataset/vectordb_bench, tar.gz文件忽略）:

```
[root@tw-node45 vectordb_bench]# tree
.
├── cohere
│   └── cohere_medium_1m
│       ├── neighbors_head_1p.parquet
│       ├── neighbors.parquet
│       ├── neighbors_tail_1p.parquet
│       ├── shuffle_train.parquet
│       ├── test.parquet
│       └── train.parquet
├── cohere_medium_1m.tar.gz
└── openai
    ├── openai_medium_500k
    │   ├── neighbors_head_1p.parquet
    │   ├── neighbors.parquet
    │   ├── neighbors_tail_1p.parquet
    │   ├── shuffle_train.parquet
    │   ├── test.parquet
    │   └── train.parquet
    ├── openai_small_50k
    │   ├── neighbors_head_1p.parquet
    │   ├── neighbors.parquet
    │   ├── neighbors_tail_1p.parquet
    │   ├── shuffle_train.parquet
    │   ├── test.parquet
    │   └── train.parquet
    └── openai_small_50k.tar.gz

```


容器里执行:

```shell
cd /opt/transwarp/vectordb_bench
python -m pip install .
init_bench
```