SHELL := /bin/sh
PYTHON ?= python3.11
PYTHONPATH_VALUE := backend:../python-package
CPU_IMAGE ?= ghcr.io/deepinsight/insightface-server:0.3.1-cpu
CUDA_IMAGE ?= ghcr.io/deepinsight/insightface-server:0.3.1-cuda12
SERVER_VERSION ?= 0.3.1
NATIVE_CPU_BUILD_DIR ?= build/native-search-cpu
DOCKER_BUILD_ARGS ?= --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
	--build-arg http_proxy --build-arg https_proxy \
	--build-arg INSIGHTFACE_SERVER_VERSION=$(SERVER_VERSION)

.PHONY: lint test test-api test-sdk test-frontend test-native-cpu update-api-docs build-cpu build-cuda12 \
	run-cpu run-cuda12 test-cpu test-cuda12 test-consistency smoke-test release-preflight \
	release-precheck

lint:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m ruff check backend tests
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m mypy backend/insightface_server

test:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m pytest tests -m "not cpu and not cuda and not consistency and not docker and not e2e"

test-api:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m pytest tests/api tests/integration

test-sdk:
	$(PYTHON) -m pytest tests/sdk

test-frontend:
	node --test tests/frontend/*.test.mjs

update-api-docs:
	$(PYTHON) scripts/update_openapi_snapshot.py

test-native-cpu:
	cmake -S native/search -B $(NATIVE_CPU_BUILD_DIR) \
		-DIFS_SEARCH_BUILD_CPU=ON \
		-DIFS_SEARCH_BUILD_CUDA=OFF \
		-DIFS_SEARCH_BUILD_TESTS=ON
	cmake --build $(NATIVE_CPU_BUILD_DIR) --parallel
	ctest --test-dir $(NATIVE_CPU_BUILD_DIR) --output-on-failure

build-cpu:
	docker build $(DOCKER_BUILD_ARGS) --platform linux/amd64 -f docker/Dockerfile.cpu -t $(CPU_IMAGE) ..

build-cuda12:
	docker build $(DOCKER_BUILD_ARGS) --platform linux/amd64 -f docker/Dockerfile.cuda12 -t $(CUDA_IMAGE) ..

run-cpu:
	docker compose -f deploy/compose.cpu.yml up -d --no-build

run-cuda12:
	docker compose -f deploy/compose.cuda12.yml up -d --no-build

test-cpu:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m pytest tests/cpu -m cpu

test-cuda12:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m pytest tests/cuda -m cuda

test-consistency:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m pytest tests/integration/test_consistency.py -m consistency

smoke-test:
	$(PYTHON) scripts/smoke_test.py --base-url http://127.0.0.1:8080

release-preflight:
	$(PYTHON) scripts/release_preflight.py --version $(SERVER_VERSION)

release-precheck:
	$(PYTHON) scripts/release_preflight.py --version $(SERVER_VERSION) --relaxed \
		--output test-results/release-preflight.json
