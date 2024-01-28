BASEDIR = 'legilsation_analysis'

default: create-requirements lint

.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: create-requirements
create-requirements:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

# Group level commands
.PHONY: get-legislation-data
get-legislation-data:
	python -m ${BASEDIR} --congress-data --scotus-data

# Individual function commands
.PHONY: congress-data
congress-data:
	python -m ${BASEDIR} --congress -v

.PHONY: scotus-data
scotus-data:
	python -m ${BASEDIR} --scotus -v

.PHONY: clean-data
clean-data:
	python -m ${BASEDIR} --clean -v

.PHONY: tokenize-data
tokenize-data:
	python -m ${BASEDIR} --tokenize -v

# run all commands
.PHONY: run
run:
	make get-legislation-data clean-data
