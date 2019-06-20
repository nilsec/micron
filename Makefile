PKG=micron
TMP_FILE:=$(shell mktemp).img

default:
	python setup.py install
	-rm -rf dist build $(PKG).egg-info

install-pip:
	pip install .
	-rm -rf dist build $(PKG).egg-info

.PHONY: install-full
install-full:
	pip install .[full]
	-rm -rf dist build $(PKG).egg-info

.PHONY: install-dev
install-de:
	pip install -e .[full]
	-rm -rf dist build $(PKG).egg-info

singularity/$(PKG).img:
	sudo singularity build $(TMP_FILE) singularity/Singularity
	cp $(TMP_FILE) singularity/$(PKG).img
	sudo rm $(TMP_FILE)

.PHONY: tests
tests: singularity/$(PKG).img
	PY_MAJOR_VERSION=py`python -c 'import sys; print(sys.version_info[0])'` pytest --cov-report term-missing -v --cov=$(PKG) --cov-config=.coveragerc tests
	flake8 $(PKG)
