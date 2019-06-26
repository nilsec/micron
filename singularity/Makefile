TAG="micron"

TMP_FILE:=$(shell mktemp).img

.PHONY: default push
default:
	cp -rp ../micron .
	cp -p ../requirements.txt .
	cp -p ../setup.py .
	sudo singularity build $(TMP_FILE) Singularity
	cp $(TMP_FILE) $(TAG).img
	sudo rm $(TMP_FILE)
