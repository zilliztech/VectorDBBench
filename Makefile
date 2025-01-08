unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench --check
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench
