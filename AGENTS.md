# Agent Instructions

## Test Execution

Run all future VectorDBBench test and benchmark commands on the remote client
machine, not on the local Mac, unless the user explicitly approves a local run.

Remote client:

- Host: `ubuntu@10.15.2.233`
- Workdir: `/home/ubuntu/VectorDBBench`
- SSH key: `/Users/james.gao/Downloads/ec2_qtp.pem`

Before running tests or benchmarks, ensure the remote client checkout is on the
latest `fts_impl_only` branch. Use the local machine for code edits and Git
orchestration only.
