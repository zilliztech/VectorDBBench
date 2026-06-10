GO ?= go
GOCACHE ?= $(CURDIR)/.cache/go-build
BIN ?= dist/s3vbench

.PHONY: test build run clean

test:
	GO111MODULE=on GOCACHE=$(GOCACHE) $(GO) test ./...

build:
	GO111MODULE=on GOCACHE=$(GOCACHE) $(GO) build -o $(BIN) ./cmd/s3vbench

run:
	GO111MODULE=on GOCACHE=$(GOCACHE) $(GO) run ./cmd/s3vbench $(ARGS)

clean:
	rm -rf .cache dist result s3vbench
