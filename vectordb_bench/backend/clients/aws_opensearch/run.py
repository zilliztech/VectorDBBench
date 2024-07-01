import time, random
from opensearchpy import OpenSearch
from opensearch_dsl import Search, Document, Text, Keyword

_HOST = 'xxxxxx.us-west-2.es.amazonaws.com'
_PORT = 443
_AUTH = ('admin', 'xxxxxx') # For testing only. Don't store credentials in code.

_INDEX_NAME = 'my-dsl-index'
_BATCH = 100
_ROWS = 100
_DIM = 128
_TOPK = 10


def create_client():
    client = OpenSearch(
        hosts=[{'host': _HOST, 'port': _PORT}],
        http_compress=True, # enables gzip compression for request bodies
        http_auth=_AUTH,
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    return client


def create_index(client, index_name):
    settings = {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "refresh_interval": "5s",
        }
    }
    mappings = {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": _DIM,
                "method": {
                    "engine": "nmslib",
                    "name": "hnsw",
                    "space_type": "l2",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 24,
                    }
                }
            }
        }
    }

    response = client.indices.create(index=index_name, body=dict(settings=settings, mappings=mappings))
    print('\nCreating index:')
    print(response)


def delete_index(client, index_name):
    response = client.indices.delete(index=index_name)
    print('\nDeleting index:')
    print(response)


def bulk_insert(client, index_name):
    # Perform bulk operations
    ids = [i for i in range(_ROWS)]
    vec = [[random.random() for _ in range(_DIM)] for _ in range(_ROWS)]

    docs = []
    for i in range(0, _ROWS, _BATCH):
        docs.clear()
        for j in range(0, _BATCH):
            docs.append({"index": {"_index": index_name, "_id": ids[i+j]}})
            docs.append({"embedding": vec[i+j]})
        response = client.bulk(docs)
        print('\nAdding documents:', len(response['items']), response['errors'])
        response = client.indices.stats(index_name)
        print('\nTotal document count in index:', response['_all']['primaries']['indexing']['index_total'])


def search(client, index_name):
    # Search for the document.
    search_body = {
        "size": _TOPK,
        "query": {
            "knn": {
                "embedding": {
                    "vector": [random.random() for _ in range(_DIM)],
                    "k": _TOPK,
                }
            }
        }
    }
    while True:
        response = client.search(index=index_name, body=search_body)
        print(f'\nSearch took: {response["took"]}')
        print(f'\nSearch shards: {response["_shards"]}')
        print(f'\nSearch hits total: {response["hits"]["total"]}')
        result = response["hits"]["hits"]
        if len(result) != 0:
            print('\nSearch results:')
            for hit in response["hits"]["hits"]:
                print(hit["_id"], hit["_score"])
            break
        else:
            print('\nSearch not ready, sleep 1s')
            time.sleep(1)


def main():
    client = create_client()
    try:
        create_index(client, _INDEX_NAME)
        bulk_insert(client, _INDEX_NAME)
        search(client, _INDEX_NAME)
        delete_index(client, _INDEX_NAME)
    except Exception as e:
        print(e)
        delete_index(client, _INDEX_NAME)


if __name__ == '__main__':
    main()
