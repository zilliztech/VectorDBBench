# Test layout

The test suite is being separated by execution requirements.

## Hermetic unit tests

`tests/unit/` is the allowlisted local suite. Tests in this directory must be
deterministic and must not use network sockets, credentials, live databases,
containers, or downloaded datasets.

Install only the project and test extra, then run:

```shell
pip install -e ".[test]"
make unit-test
```

`make unittest` is a compatibility alias. The unit target uses
`pytest-socket` to disable network access and is the suite run by pull-request
CI.

## Online E2E tests

`tests/e2e/` contains tests that download datasets or require external
services. Run them explicitly with:

```shell
make e2e-test
```

The dataset E2E tests require network access to S3 and/or Aliyun OSS and can
download large Cohere or LAION datasets. Other E2E tests may require provider
SDK extras, credentials, containers, or a locally running database; document
those prerequisites with the test when migrating it.

## Legacy tests

Test modules directly under `tests/` are legacy and are not yet classified.
They are intentionally outside both Make targets: do not assume they are
hermetic, and do not move them into `tests/unit/` until they pass with only
`.[test]` and with network sockets disabled. New tests must go directly into
`tests/unit/` or `tests/e2e/`.
