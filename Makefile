define HELP

This is the project Makefile.

Usage:

make black                   - Run code formatter
make lint                    - Run linter
make requirements            - Install dev/testing requirements
make clean                   - Remove generated files

endef

export HELP


help:
	@echo "$$HELP"


.PHONY: requirements
requirements: .requirements.txt


.requirements.txt: requirements/development.txt
	pip install -r requirements/development.txt


.PHONY: black
black: requirements
	black src


.PHONY: lint
lint: requirements
	flake8 . | tee lint.txt


.PHONY: clean
clean:
	rm -rf .coverage coverage.xml .requirements.txt *.zip .cache .pytest_cache
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete