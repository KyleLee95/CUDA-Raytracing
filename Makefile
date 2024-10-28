# Makefile

CC = g++
NVCC = nvcc
CFLAGS = -I./include
CUDAFLAGS = -I./include

# Build the scene generator
scene_generator: src/scene_generator.cpp
	$(CC) $(CFLAGS) -o scene_generator src/scene_generator.cpp

# Build the sequential ray tracer
sequential: main.cpp src/sequential_raytracer.cpp src/utils.cpp
	$(CC) $(CFLAGS) -o sequential_raytracer main.cpp src/sequential_raytracer.cpp src/utils.cpp

# Build compare images
compare_images: src/compare_images.cpp src/utils.cpp
	$(CC) $(CFLAGS) -o compare_images src/compare_images.cpp src/utils.cpp

# Build the CUDA ray tracer
cuda: main.cu src/cuda_raytracer.cu src/utils.cpp
	$(NVCC) $(CUDAFLAGS) -o cuda_raytracer main.cu src/cuda_raytracer.cu src/utils.cpp

# Build all
all: scene_generator sequential cuda compare_images

clean:
	rm -rf scene_generator sequential_raytracer cuda_raytracer compare_images execution_timings.png output

