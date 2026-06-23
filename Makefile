.PHONY: web cli install selfcheck

install:
	pip install -r requirements.txt

web:        ## launch the browser app at http://127.0.0.1:8000
	python web/app.py

cli:        ## launch the terminal runner
	python practice.py

selfcheck:  ## verify every reference solution passes its own tests
	@bash scripts/selfcheck.sh
