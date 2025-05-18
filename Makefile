.PHONY: lint test

lint:
	pysen run lint

test:
	pytest tests
