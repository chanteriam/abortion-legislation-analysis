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
	python -m legilsation_analysis --congress-data --scotus-data

# Individual function commands
.PHONY: congress-data
congress-data:
	python -m legilsation_analysis --congress -v

.PHONY: scotus-data
scotus-data:
	python -m legilsation_analysis --scotus -v

.PHONY: clean-data
clean-data:
	python -m legilsation_analysis --clean -v

.PHONY: tokenize-data
tokenize-data:
	python -m legilsation_analysis --tokenize -v

# run all commands
.PHONY: run
run:
	make get-legislation-data clean-data
