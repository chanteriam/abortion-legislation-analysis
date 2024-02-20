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
	python -m legislation_analysis --congress-data --scotus-data

# Individual function commands
.PHONY: congress-data
congress-data:
	python -m legislation_analysis --congress -d

.PHONY: scotus-data
scotus-data:
	python -m legislation_analysis --scotus -d

.PHONY: clean-data
clean-data:
	python -m legislation_analysis --clean -d

.PHONY: tokenize-data
tokenize-data:
	python -m legislation_analysis --tokenize -d

.PHONY: pos-tag-data
pos-tag-data:
	python -m legislation_analysis --pos-tag -d

.PHONY: cluster-data
cluster-data:
	python -m legislation_analysis --cluster -d

# run all commands
.PHONY: run
run:
	make get-legislation-data --all
