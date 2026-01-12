


PWD=$(shell pwd)
PYTHON=$(PWD)/python
PYCACHE=$(PYTHON)/__pycache__


clean:
	rm $(PYCACHE) -rf
	rm ./log -rf