# Test pyKnowhere with VectorDBBench

make sure python >= 3.11

## Git Clone Knowhere
```sh
git clone https://github.com/zilliztech/knowhere.git
cd knowhere
```

## Install Dependency for Knowhere (C++)
```sh
sudo apt update \
&& sudo apt install -y cmake g++ gcc libopenblas-dev libaio-dev libcurl4-openssl-dev libevent-dev libgflags-dev python3 python3-pip python3-setuptools \
&& pip3 install conan==1.61.0 pytest faiss-cpu numpy wheel \
&& pip3 install bfloat16 \
&& conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
```

## Build Knowhere 
```sh
rm -rf build \
&& mkdir build \
&& cd build \
&& conan install .. --build=missing -o with_raft=True -s compiler.libcxx=libstdc++11 -s build_type=Release \
&& conan build ..
&& cd ..
```

## Swig and Install (Python)
```sh
cd python \
&& rm -rf dist \
&& python3 setup.py bdist_wheel \
&& pip3 install --force-reinstall dist/*.whl \
&& cd ..
```

## Clone and Install VectorDBBench 
```sh
git clone -b library_test_pyknowhere https://github.com/zilliztech/VectorDBBench.git
cd VectorDBBench
pip install -e ".[test]"
```

## Run Test
```sh
init_bench
```
After running it you will get a url and open it.

The default page show all the `database` test results, but pyknowhere is `library` type.

These sub-pages may help you.
- `/run_test` select the test client and cases to run test.
- `/library_results` show all the `library` test results.

Note that, VDBBench does **not yet support filtering tests and capacity tests** for library type clients.

Some config examples of pyknowhere GPU_Index
- GPU_IVF_FLAT
  - "nlist": 1024, "nprobe": 64, "cache_dataset_on_device": "false", "refine_ratio": 1.0
- GPU_IVF_PQ
  - "nlist": 1024, "nprobe": 64, "nbits": 8, "m": 0, "cache_dataset_on_device": "false", "refine_ratio": 1.0
- GPU_CAGRA
  - "intermediate_graph_degree": 64, "graph_degree": 32, "build_algo": "IVF_PQ", "cache_dataset_on_device": "false", "itopk_size": 128, "team_size": 0, "search_width": 4, "min_iterations": 0, "max_iterations": 0, "refine_ratio": 1.0
  - `build_algo`: `"NN_DESCENT"` or `"IVF_PQ"`
