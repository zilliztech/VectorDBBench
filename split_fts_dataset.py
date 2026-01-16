import pathlib


# --- 配置区域：在这里修改参数 ---
# DATASET_DIR: 数据集所在的目录路径
DATASET_DIR = pathlib.Path("dataset")

# SIZES: 你想要生成的子集规模列表（文档数量）
# 例如 [100000, 500000] 会生成 100k 和 500k 的数据集
SIZES = [1000000] 
# ------------------------------

def format_size_label(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)

def split_one(max_docs: int) -> None:
    root = DATASET_DIR.resolve()
    collection_src = root / "collectionfull.tsv"
    qrels_src = root / "qrelsfull.dev.tsv"
    queries_src = root / "queriesfull.dev.tsv"

    label = format_size_label(max_docs)
    collection_out = root / f"collection.{label}.tsv"
    qrels_out = root / f"qrels.dev.{label}.tsv"
    queries_out = root / f"queries.dev.{label}.tsv"

    doc_ids: set[int] = set()
    count_docs = 0
    with collection_src.open("r", encoding="utf-8") as fin, collection_out.open(
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            try:
                doc_id = int(parts[0])
            except ValueError:
                continue
            fout.write(line)
            doc_ids.add(doc_id)
            count_docs += 1
            if count_docs >= max_docs:
                break

    qids: set[int] = set()
    count_qrels = 0
    with qrels_src.open("r", encoding="utf-8") as fin, qrels_out.open(
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                query_id = int(parts[0])
                doc_id = int(parts[2])
            except ValueError:
                continue
            if doc_id not in doc_ids:
                continue
            fout.write(line)
            qids.add(query_id)
            count_qrels += 1

    count_queries = 0
    with queries_src.open("r", encoding="utf-8") as fin, queries_out.open(
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            try:
                query_id = int(parts[0])
            except ValueError:
                continue
            if query_id not in qids:
                continue
            fout.write(line)
            count_queries += 1

    print(
        f"size={max_docs} ({label}): docs={count_docs}, qrels={count_qrels}, queries={count_queries}",
    )


def main() -> None:
    for size in SIZES:
        split_one(size)


if __name__ == "__main__":
    main()
