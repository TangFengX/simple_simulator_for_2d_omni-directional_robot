


PWD:=$(shell pwd)
PYTHON:=$(PWD)/python
PYCACHE:=$(PYTHON)/__pycache__

CC      := gcc
CFLAGS  := -O2 -fPIC -Wall -Wextra -Iinclude
LDFLAGS := -shared
TARGET  := bin/libpilot.so
SRCS := $(wildcard src/*.c)
OBJS := $(patsubst src/%.c, build/%.o, $(SRCS))

$(TARGET): $(OBJS)
	@mkdir -p bin
	$(CC) $(LDFLAGS) -o $@ $^

build/%.o: src/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

debug:
	$(MAKE) CFLAGS="-g -O0 -fPIC -Wall" all

clean:
	rm $(PYCACHE) -rf
	rm ./log -rf
	rm ./build -rf
	rm ./bin -rf