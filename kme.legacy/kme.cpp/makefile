# MAKEFILE FOR
# MACOS ARCHITECTURE

CC=g++

# INCLUDE: directories for header files
INCLUDE=-I/opt/homebrew/include

# LIB: libraries to link against
LIB=-L/opt/homebrew/lib -lglfw -lboost_program_options -lboost_filesystem

# FLAGS: Compiler flags for all targets
# -std=c++17: Use the C++17 standard
# -Wall: Enable all compiler warnings
FLAGS=-std=c++17 -Wall

# LIBFLAGS: Additional flags specific to the rewarder target (shared library)
# -fPIC: Generate position-independent code, necessary for shared libraries
# -shared: Create a shared library instead of an executable
# -Wl,-install_name,librewarder.so: Set the shared library's soname to 'librewarder.so'
LIBFLAGS=-fPIC -shared -Wl,-install_name,librewarder.so

# Source files for rewarder target
FILES_REW = src/util/sampler.cpp \
                src/util/util.cpp \
                src/encoder/kmeans_encoder.cpp \
                src/rewarder/kmeans_rewarder.cpp \
                kme/rewarder.c


all: rewarder

rewarder:
	$(CC) $(INCLUDE) $(LIB) $(FLAGS) $(LIBFLAGS) $(FILES_REW) -o kme/librewarder.so 

.PHONY: clean

clean:
	rm -r build entropy kme/librewarder.so