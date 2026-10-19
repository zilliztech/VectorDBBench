# Hugging Face Dataset Support

September 2026

VectorDBBench can now run registered vector-search datasets hosted on Hugging
Face. This release adds all 24 datasets advertised by VIBE and three VDBBench
multimodal embedding datasets without introducing a new benchmark case or a
source-specific public interface.

## User interface

Select a registered dataset by name through the existing `Performance` case:

```shell
vectordbbench milvushnsw \
  --case-type Performance \
  --dataset-name glove-200-cosine \
  --k 100
```

The same dataset registrations appear in the web interface under `VIBE Search
Performance` and `VDBBench Multimodal Search Performance`. The database command
and its index and search parameters are otherwise unchanged.

## Architecture

The dataset source and storage format are independent concerns:

```text
Performance + --dataset-name
             |
             v
      dataset registry
             |
             v
   DatasetManager implementation
       HDF5 or Parquet
             |
             v
      DatasetSource reader
Hugging Face / S3 / Aliyun OSS
             |
             v
        cached local artifacts
        queries + GT in memory
        corpus streamed in batches
             |
             v
      common runner contract
```

`HuggingFaceReader` resolves files from a repository at a pinned revision. The
selected `Hdf5DatasetManager` or `ParquetDatasetManager` validates and exposes
those files through the same batch iterator and query/ground-truth interface
already consumed by benchmark runners. Existing vector datasets continue to
use their configured S3 or Aliyun OSS source, and full-text datasets continue
to use `ir_datasets`.

## VIBE catalog

The [VIBE repository](https://huggingface.co/datasets/vector-index-bench/vibe)
publishes each dataset as one HDF5 artifact. VectorDBBench registers all 24
advertised artifacts, including the five that the upstream catalog marks as
deprecated. Normalized artifacts use `COSINE` in VectorDBBench; explicitly
Euclidean and inner-product artifacts use `L2` and `IP`, respectively.

| Dataset name | Distribution | Modality | Corpus vectors | Dimensions | Metric | Upstream status |
|---|---|---|---:|---:|---|---|
| `agnews-mxbai-1024-euclidean` | ID | Text | 769,382 | 1,024 | L2 | Current |
| `arxiv-nomic-768-normalized` | ID | Text | 1,344,643 | 768 | COSINE | Current |
| `dpr-jina-768-normalized` | ID | Text | 20,969,760 | 768 | COSINE | Current |
| `glove-200-cosine` | ID | Word | 1,192,514 | 200 | COSINE | Current |
| `gooaq-distilroberta-768-normalized` | ID | Text | 1,475,024 | 768 | COSINE | Current |
| `imagenet-clip-512-normalized` | ID | Image | 1,281,167 | 512 | COSINE | Current |
| `inaturalist-resnet-2048-cosine` | ID | Image | 499,000 | 2,048 | COSINE | Current |
| `landmark-dino-768-cosine` | ID | Image | 760,757 | 768 | COSINE | Current |
| `landmark-nomic-768-normalized` | ID | Image | 760,757 | 768 | COSINE | Current |
| `msmarco-qwen-1024-normalized` | ID | Text | 8,840,823 | 1,024 | COSINE | Current |
| `yahoo-minilm-384-normalized` | ID | Text | 677,305 | 384 | COSINE | Current |
| `hotpotqa-harrier-640-normalized` | OOD | Text | 5,233,329 | 640 | COSINE | Current |
| `imagenet-align-640-normalized` | OOD | Text-to-Image | 1,281,167 | 640 | COSINE | Current |
| `laion-clip-512-normalized` | OOD | Text-to-Image | 1,000,448 | 512 | COSINE | Current |
| `yandex-200-cosine` | OOD | Text-to-Image | 1,000,000 | 200 | COSINE | Current |
| `cqadupstack-lemur-2048-ip` | OOD | Multi-vector encoding | 457,149 | 2,048 | IP | Current |
| `cqadupstack-muvera-5120-ip` | OOD | Multi-vector encoding | 457,149 | 5,120 | IP | Current |
| `yi-128-ip` | OOD | Attention | 187,843 | 128 | IP | Current |
| `llama-128-ip` | OOD | Attention | 256,921 | 128 | IP | Current |
| `ccnews-nomic-768-normalized` | ID | Text | 495,328 | 768 | COSINE | Deprecated |
| `celeba-resnet-2048-cosine` | ID | Image | 201,599 | 2,048 | COSINE | Deprecated |
| `coco-nomic-768-normalized` | OOD | Text-to-Image | 282,360 | 768 | COSINE | Deprecated |
| `codesearchnet-jina-768-cosine` | ID | Code | 1,374,067 | 768 | COSINE | Deprecated |
| `simplewiki-openai-3072-normalized` | ID | Text | 260,372 | 3,072 | COSINE | Deprecated |

VIBE HDF5 files provide `train`, `test`, `neighbors`, and `distances` arrays.
The published ground truth contains 100 neighbors per query. OOD artifacts can
also contain learning-query arrays, but the initial integration uses the
canonical `test` queries and their `neighbors` ground truth.

## VDBBench multimodal catalog

The VDBBench organization publishes the same multimodal embedding workload at
three corpus sizes. All three use 4,096-dimensional, L2-normalized `float32`
embeddings, inner-product search, 10,000 queries, and top-100 ground truth.

| Dataset name | Repository | Corpus vectors | Corpus layout |
|---|---|---:|---|
| `multimodal-embedding-1m` | [`VDBBench/multimodal-embedding-1M`](https://huggingface.co/datasets/VDBBench/multimodal-embedding-1M) | 1,000,000 | Single Parquet file |
| `multimodal-embedding-10m` | [`VDBBench/multimodal-embedding-10M`](https://huggingface.co/datasets/VDBBench/multimodal-embedding-10M) | 10,000,000 | Flat Parquet shards |
| `multimodal-embedding-100m` | [`VDBBench/multimodal-embedding-100M`](https://huggingface.co/datasets/VDBBench/multimodal-embedding-100M) | 100,000,000 | Nested Parquet shards |

The 10M and 100M registrations use explicit wildcard selectors for their
corpus and query shards. Ground-truth files remain separately identified, so
file roles do not depend on a repository having one particular directory
layout.

## Download and memory behavior

Dataset preparation resolves every required artifact before timed insertion
begins. Exact filenames use `hf_hub_download`; wildcard selectors use
`snapshot_download` with `allow_patterns`. Both paths use the standard
[`huggingface_hub` cache, authentication, and revision behavior](https://huggingface.co/docs/huggingface_hub/en/guides/download).
Credentials are not dataset case parameters and are not written to benchmark
results.

The format managers then apply different read strategies behind the common
runner interface:

- HDF5 query and ground-truth arrays are loaded into memory during preparation.
  The corpus is not converted or materialized in memory. Each insertion worker
  opens the HDF5 file once for the lifetime of its iterator, reads bounded
  batches, and closes the file when iteration finishes.
- Parquet corpus selectors are expanded and sorted deterministically. Query
  shards are concatenated in that order, ground truth is validated against the
  query IDs and requested width, and corpus files remain batch-streamed through
  the existing Parquet iterator.

Result JSONs can include dataset provenance such as the registered name,
family, source repository, pinned revision, metric, point type, storage format,
and resolved role files. The field is optional so existing result files remain
readable.

## Constraints and operational notes

- These registered Hugging Face cases are unfiltered. Their artifacts do not
  provide the scalar fields and filtered ground truth required for filtered
  benchmark cases.
- Search K must be between 1 and 100 because the registered artifacts publish
  top-100 ground truth.
- Downloads use the local Hugging Face cache and can require substantial disk
  space. In particular, the
  [100M multimodal repository](https://huggingface.co/datasets/VDBBench/multimodal-embedding-100M)
  reports approximately 1.51 TB of files.
- The initial development verification exercised three smaller VIBE datasets
  and the 1M VDBBench multimodal dataset against Milvus. The 10M and 100M
  registrations are covered by manifest, selector, schema, and fixture tests;
  the complete 100M download was not part of that verification.

## Licensing and provenance

VectorDBBench downloads artifacts from their upstream Hugging Face repositories
and does not bundle or mirror them. The VIBE dataset card marks the repository
as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and separately
credits [GloVe](https://nlp.stanford.edu/projects/glove/) under
[PDDL 1.0](https://opendatacommons.org/licenses/pddl/1-0/) plus
[LAION](https://laion.ai/blog/laion-400-open-dataset/) and
[Yandex](https://big-ann-benchmarks.com/neurips23.html) subsets under CC BY
4.0. See the [VIBE paper](https://arxiv.org/abs/2505.17810),
[dataset card](https://huggingface.co/datasets/vector-index-bench/vibe), and
linked upstream notices before redistributing or adapting those artifacts.

The VDBBench multimodal dataset cards describe their LAION-derived source,
embedding generation, license, and attribution requirements. Users remain
responsible for checking the applicable dataset card and upstream terms for
their intended use.
