.PHONY: lint

lint:
	black kme.py && autoflake -i -r --remove-all-unused-imports kme.py