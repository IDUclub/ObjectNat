CODE := src
PACKAGE := objectnat

.PHONY: help lint format test docs clean build publish build-and-publish \
        install install-dev update release version-patch version-minor version-major tag push-tag

help:
	@echo "Available targets:"
	@echo "  lint              - run linters (pylint, isort, black --check)"
	@echo "  format            - format code with isort + black"
	@echo "  test              - run test suite (pytest)"
	@echo "  docs              - build documentation"
	@echo "  build             - build wheel and sdist via poetry"
	@echo "  publish           - publish to PyPI via poetry"
	@echo "  build-and-publish - clean + build + publish"
	@echo "  release           - tag current version (from pyproject) and push tag"
	@echo "  version-patch     - bump patch version via poetry"
	@echo "  version-minor     - bump minor version via poetry"
	@echo "  version-major     - bump major version via poetry"

lint:
	poetry run pylint $(CODE)
	poetry run isort --check-only $(CODE)
	poetry run black --check $(CODE)

format:
	poetry run isort $(CODE)
	poetry run black $(CODE)

test:
	poetry run pytest

docs:
	poetry run sphinx-build -b html docs docs/_build/html

install:
	pip install .

install-dev:
	poetry install --with dev,test,docs

clean:
	rm -rf ./dist

build:
	poetry build

publish:
	poetry publish

build-and-publish: clean build publish

update:
	poetry update

.PHONY: sync-version-file
sync-version-file:
	python scripts/sync_version.py

.PHONY: version-patch version-minor version-major
version-patch: # 0.0.v
	poetry version patch
	$(MAKE) sync-version-file

version-minor: # 0.v.0
	poetry version minor
	$(MAKE) sync-version-file

version-major: # v.0.0
	poetry version major
	$(MAKE) sync-version-file

VERSION := $(shell poetry version -s)

tag:
	git tag v$(VERSION)
	@echo "Created tag v$(VERSION)"

push-tag:
	git push origin v$(VERSION)

release: tag push-tag
	@echo "Tagged and pushed v$(VERSION)."
