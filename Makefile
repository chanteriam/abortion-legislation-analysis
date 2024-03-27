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
	python -m legislation_analysis --congress --scotus -d

# Individual function commands
.PHONY: get-congress-data
get-congress-data:
	python -m legislation_analysis --congress -d

.PHONY: get-scotus-data
get-scotus-data:
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

.PHONY: ner-data
ner-data:
	python -m legislation_analysis --ner -d

.PHONY: cluster-data
cluster-data:
	python -m legislation_analysis --cluster -d

.PHONY: model-data
model-data:
	python -m legislation_analysis --model -d

.PHONY: network-data
network-data:
	python -m legislation_analysis --network -d

# run all commands
.PHONY: run
run:
	make get-legislation-data --all
