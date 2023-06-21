# VectorDBBench: A Benchmark Tool for VectorDB

[![version](https://img.shields.io/pypi/v/vectordb-bench.svg?color=blue)](https://pypi.org/project/vectordb-bench/)
[![Downloads](https://pepy.tech/badge/vectordb-bench)](https://pepy.tech/project/vectordb-bench)

## Quick Start
### Prerequirement
``` shell
python >= 3.11
```
### Install
``` shell
pip install vectordb-bench
```
### Run

``` shell
init_bench
```
## What is VectorDBBench
VectorDBBench is not just an offering of benchmark results for mainstream vector databases and cloud services, it's your go-to tool for the ultimate performance and cost-effectiveness comparison. Designed with ease-of-use in mind, VectorDBBench is devised to help users, even non-professionals, reproduce results or test new systems, making the hunt for the optimal choice amongst a plethora of cloud services and open-source vector databases a breeze.

Understanding the importance of user experience, we provide an intuitive visual interface. This not only empowers users to initiate benchmarks at ease, but also to view comparative result reports, thereby reproducing benchmark results effortlessly.
To add more relevance and practicality, we provide cost-effectiveness reports particularly for cloud services. This allows for a more realistic and applicable benchmarking process.

Closely mimicking real-world production environments, we've set up diverse testing scenarios including insertion, searching, and filtered searching. To provide you with credible and reliable data, we've included public datasets from actual production scenarios, such as [SIFT](http://corpus-texmex.irisa.fr/), [GIST](http://corpus-texmex.irisa.fr/), [Cohere](https://huggingface.co/datasets/Cohere/wikipedia-22-12/tree/main/en), and more. It's fascinating to discover how a relatively unknown open-source database might excel in certain circumstances!

Prepare to delve into the world of VectorDBBench, and let it guide you in uncovering your perfect vector database match.  

## Build on your own
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
### Check coding styles
```shell
$ ruff check vectordb_bench
```

Add `--fix` if you want to fix the coding styles automatically

```shell
$ ruff check vectordb_bench --fix
```

## How does it work?
### Result Page
![image](https://github.com/zilliztech/VectorDBBench/assets/105927039/66ab83c4-656e-41a8-a643-d9790faccbeb)
This is the main page of VectorDBBench, which displays the standard benchmark results we provide. Additionally, results of all tests performed by users themselves will also be shown here. We also offer the ability to select and compare results from multiple tests simultaneously.

The standard benchmark results displayed here include all 9 cases that we currently support for all our clients (Milvus, Zilliz Cloud, Elastic Search, Qdrant Cloud, and Weaviate Cloud). However, as some systems may not be able to complete all the tests successfully due to issues like Out of Memory (OOM) or timeouts, not all clients are included in every case.
### Run Test Page
![image](https://github.com/zilliztech/VectorDBBench/assets/105927039/a789099a-3707-4214-8052-b73463b8f2c6)
This is the page to run a test:
1. Initially, you select the systems to be tested - multiple selections are allowed. Once selected, corresponding forms will pop up to gather necessary information for using the chosen databases. The db_label is used to differentiate different instances of the same system. We recommend filling in the host size or instance type here (as we do in our standard results).
2. The next step is to select the test cases you want to perform. You can select multiple cases at once, and a form to collect corresponding parameters will appear.
3. Finally, you'll need to provide a task label to distinguish different test results. Using the same label for different tests will result in the previous results being overwritten.
Now we can only run one task at the same time. 

## Module
### Code Structure
![image](https://github.com/zilliztech/VectorDBBench/assets/105927039/8c06512e-5419-4381-b084-9c93aed59639)
### Client
Our client module is designed with flexibility and extensibility in mind, aiming to integrate APIs from different systems seamlessly. As of now, it supports Milvus, Zilliz Cloud, Elastic Search, Pinecone, Qdrant, and Weaviate. Stay tuned for more options, as we are consistently working on extending our reach to other systems.
### Benchmark Cases
We've developed an array of 9 comprehensive benchmark cases to test vector databases' various capabilities, each designed to give you a different piece of the puzzle. These cases are categorized into three main types:
#### Capacity Case
- **Large Dim:** Tests the database's loading capacity by inserting large-dimension vectors (GIST 100K vectors, 960 dimensions) until fully loaded. The final number of inserted vectors is reported.
- **Small Dim:** Similar to the Large Dim case but uses small-dimension vectors (SIFT 100K vectors, 128 dimensions).
#### Search Performance Case
- **XLarge Dataset:** Measures search performance with a massive dataset (LAION 100M vectors, 768 dimensions) at varying parallel levels. The results include index building time, recall, latency, and maximum QPS.
- **Large Dataset:** Similar to the XLarge Dataset case, but uses a slightly smaller dataset (Cohere 10M vectors, 768 dimensions).
- **Medium Dataset:** A case using a medium dataset (Cohere 1M vectors, 768 dimensions).
#### Filtering Search Performance Case
- **Large Dataset, Low Filtering Rate:** Evaluates search performance with a large dataset (Cohere 10M vectors, 768 dimensions) under a low filtering rate (1% vectors) at different parallel levels.
- **Medium Dataset, Low Filtering Rate:** This case uses a medium dataset (Cohere 1M vectors, 768 dimensions) with a similar low filtering rate.
- **Large Dataset, High Filtering Rate:** It tests with a large dataset (Cohere 10M vectors, 768 dimensions) but under a high filtering rate (99% vectors).
- **Medium Dataset, High Filtering Rate:** This case uses a medium dataset (Cohere 1M vectors, 768 dimensions) with a high filtering rate.
For a quick reference, here is a table summarizing the key aspects of each case:

Case No. | Case Type | Dataset Size | Dataset Type | Filtering Rate | Results |
|----------|-----------|--------------|--------------|----------------|---------|
1 | Capacity Case | Large Dim | GIST 100K vectors, 960 dimensions | N/A | Number of inserted vectors |
2 | Capacity Case | Small Dim | SIFT 100K vectors, 128 dimensions | N/A | Number of inserted vectors |
3 | Search Performance Case | XLarge Dataset | LAION 100M vectors, 768 dimensions | N/A | Index building time, recall, latency, maximum QPS |
4 | Search Performance Case | Large Dataset | Cohere 10M vectors, 768 dimensions | N/A | Index building time, recall, latency, maximum QPS |
5 | Search Performance Case | Medium Dataset | Cohere 1M vectors, 768 dimensions | N/A | Index building time, recall, latency, maximum QPS |
6 | Filtering Search Performance Case | Large Dataset, Low Filtering Rate | Cohere 10M vectors, 768 dimensions | 1% vectors | Index building time, recall, latency, maximum QPS |
7 | Filtering Search Performance Case | Medium Dataset, Low Filtering Rate | Cohere 1M vectors, 768 dimensions | 1% vectors | Index building time, recall, latency, maximum QPS |
8 | Filtering Search Performance Case | Large Dataset, High Filtering Rate | Cohere 10M vectors, 768 dimensions | 99% vectors | Index building time, recall, latency, maximum QPS |
9 | Filtering Search Performance Case | Medium Dataset, High Filtering Rate | Cohere 1M vectors, 768 dimensions | 99% vectors | Index building time, recall, latency, maximum QPS |

Each case provides an in-depth examination of a vector database's abilities, providing you a comprehensive view of the database's performance.

## Goals
Our goals of this benchmark are:
### Reproducibility & Usability
One of the primary goals of VectorDBBench is to enable users to reproduce benchmark results swiftly and easily, or to test their customized scenarios. We believe that lowering the barriers to entry for conducting these tests will enhance the community's understanding and improvement of vector databases. We aim to create an environment where any user, regardless of their technical expertise, can quickly set up and run benchmarks, and view and analyze results in an intuitive manner.
### Representability & Realism
VectorDBBench aims to provide a more comprehensive, multi-faceted testing environment that accurately represents the complexity of vector databases. By moving beyond a simple speed test for algorithms, we hope to contribute to a better understanding of vector databases in real-world scenarios. By incorporating as many complex scenarios as possible, including a variety of test cases and datasets, we aim to reflect realistic conditions and offer tangible significance to our community. Our goal is to deliver benchmarking results that can drive tangible improvements in the development and usage of vector databases.

## Contribution
### General Guidelines
1. Fork the repository and create a new branch for your changes.
2. Adhere to coding conventions and formatting guidelines.
3. Use clear commit messages to document the purpose of your changes.
### Adding New Clients
**Step 1: Creating New Client Files**

1. Navigate to the vectordb_bench/backend/clients directory.
2. Create a new folder for your client, for example, "new_client".
3. Inside the "new_client" folder, create two files: new_client.py and config.py.

**Step 2: Implement new_client.py and config.py**

1. Open new_client.py and define the NewClient class, which should inherit from the clients/api.py file's VectorDB abstract class. The VectorDB class serves as the API for benchmarking, and all DB clients must implement this abstract class. 
Example implementation in new_client.py:
new_client.py
```python 
from ..api import VectorDB
class NewClient(VectorDB):
    # Implement the abstract methods defined in the VectorDB class
    # ...
```
2. Open config.py and implement the DBConfig and optional DBCaseConfig classes.
  1. The DBConfig class should be an abstract class that provides information necessary to establish connections with the database. It is recommended to use the pydantic.SecretStr data type to handle sensitive data such as tokens, URIs, or passwords.
  2. The DBCaseConfig class is optional and allows for providing case-specific configurations for the database. If not provided, it defaults to EmptyDBCaseConfig.
Example implementation in config.py:
```python
from pydantic import SecretStr
from clients.api import DBConfig, DBCaseConfig

class NewDBConfig(DBConfig):
    # Implement the required configuration fields for the database connection
    # ...
    token: SecretStr
    uri: str

class NewDBCaseConfig(DBCaseConfig):
    # Implement optional case-specific configuration fields
    # ...
```
**Step 3: Importing the DB Client and Updating Initialization**

In this final step, you will import your DB client into clients/__init__.py and update the initialization process.
1. Open clients/__init__.py and import your NewClient from new_client.py.
2. Add your NewClient to the DB enum. 
3. Update the db2client dictionary by adding an entry for your NewClient.
Example implementation in clients/__init__.py:
```python
#clients/__init__.py

from .new_client.new_client import NewClient

#Add NewClient to the DB enum
class DB(Enum):
    ...
    DB.NewClient = "NewClient"

#Add NewClient to the db2client dictionary
db2client = {
    DB.Milvus: Milvus,
    ...
    DB.NewClient: NewClient
}
```
That's it! You have successfully added a new DB client to the vectordb_bench project.

## Rules
### Installation 
The system under test can be installed in any form to achieve optimal performance. This includes but is not limited to binary deployment, Docker, and cloud services.
### Fine-Tuning
For the system under test, we use the default server-side configuration to maintain the authenticity and representativeness of our results.
For the Client, we welcome any parameter tuning to obtain better results.
### Incomplete Results
Many databases may not be able to complete all test cases due to issues such as Out of Memory (OOM), crashes, or timeouts. In these scenarios, we will clearly state these occurrences in the test results.
### Mistake Or Misrepresentation 
We strive for accuracy in learning and supporting various vector databases, yet there might be oversights or misapplications. For any such occurrences, feel free to [raise an issue](https://github.com/zilliztech/VectorDBBench/issues/new) or make amendments on our GitHub page.
