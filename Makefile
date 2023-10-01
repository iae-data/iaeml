.PHONY: install test docs clean

install:
	poetry install

test:
	poetry run pytest

docs:
	$(MAKE) -C docs html

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	$(MAKE) -C docs clean
