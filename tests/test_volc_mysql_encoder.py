import struct

from vectordb_bench.backend.clients.volc_mysql.volc_mysql import _encode_batch_to_tsv


def test_encode_batch_to_tsv_sorts_by_id_and_hex_encodes(tmp_path):
    tsv = tmp_path / "batch.tsv"
    _encode_batch_to_tsv(
        metadata=[42, 7, 99],
        embeddings=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dim=2,
        tsv_path=str(tsv),
    )

    lines = tsv.read_text().splitlines()
    assert len(lines) == 3

    # Sorted by id ascending: 7, 42, 99
    assert lines[0].startswith("7\t")
    assert lines[1].startswith("42\t")
    assert lines[2].startswith("99\t")

    # The (3.0, 4.0) embedding was paired with id=7
    _, hex_str = lines[0].split("\t")
    floats = struct.unpack("<2f", bytes.fromhex(hex_str))
    assert floats == (3.0, 4.0)


def test_encode_batch_to_tsv_writes_dim_floats_per_row(tmp_path):
    tsv = tmp_path / "batch.tsv"
    _encode_batch_to_tsv(
        metadata=[1],
        embeddings=[[0.5] * 1536],
        dim=1536,
        tsv_path=str(tsv),
    )

    line = tsv.read_text().rstrip("\n")
    id_str, hex_str = line.split("\t")
    assert id_str == "1"
    # 1536 floats * 4 bytes * 2 hex chars = 12288 chars
    assert len(hex_str) == 12288
    assert bytes.fromhex(hex_str) == struct.pack("<1536f", *([0.5] * 1536))


def test_encode_batch_to_tsv_empty_batch(tmp_path):
    tsv = tmp_path / "batch.tsv"
    _encode_batch_to_tsv(metadata=[], embeddings=[], dim=4, tsv_path=str(tsv))
    assert tsv.read_text() == ""


def test_encode_batch_to_tsv_text_mode_sorts_and_formats(tmp_path):
    # binary=False (to_vector fallback): rows sorted by id, vector written as a
    # delimiter-safe "[f1,f2,...]" literal with no tab/newline inside the field.
    tsv = tmp_path / "batch.tsv"
    _encode_batch_to_tsv(
        metadata=[42, 7, 99],
        embeddings=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dim=2,
        tsv_path=str(tsv),
        binary=False,
    )

    lines = tsv.read_text().splitlines()
    assert len(lines) == 3
    # Sorted by id ascending: 7, 42, 99; id=7 was paired with (3.0, 4.0)
    assert lines[0] == "7\t[3.0,4.0]"
    assert lines[1] == "42\t[1.0,2.0]"
    assert lines[2] == "99\t[5.0,6.0]"
    # field is delimiter-safe: no tab/newline inside the bracketed literal
    for line in lines:
        _id, vec = line.split("\t")
        assert vec.startswith("[") and vec.endswith("]")
        assert "\t" not in vec
