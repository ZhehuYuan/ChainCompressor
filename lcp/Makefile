PYTHON ?= python3
CXX ?= g++
CXXFLAGS ?= -O3
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PYBIND11_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes)
TARGET := lcps$(EXT_SUFFIX)
.DEFAULT_GOAL := all
.PHONY: all check-deps clean
all: $(TARGET)
check-deps:
	$(PYTHON) -c 'import pybind11, numpy'
$(TARGET): lcps.cpp Makefile | check-deps
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -shared -fPIC -fvisibility=hidden \
		$(PYBIND11_INCLUDES) $(LDFLAGS) lcps.cpp -o "$@" $(LDLIBS)
clean:
	$(RM) "$(TARGET)"
