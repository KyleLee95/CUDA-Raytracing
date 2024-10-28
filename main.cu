#include "cuda_raytracer.h"
#include "raytracer.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(error)                                                      \
  if (error != cudaSuccess) {                                                  \
    printf("Fatal error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__,  \
           __LINE__);                                                          \
    exit(1);                                                                   \
  }

struct Arguments {
  std::string sceneFile;
  int blockX;
  int blockY;
  int width;
  int height;
  bool umem;
  int sphereCount;
};

bool parseArguments(int argc, char **argv, Arguments &args) {
  args.blockX = 16; // Default block size
  args.blockY = 16;
  args.width = 800;  // Default width
  args.height = 600; // Default height
  args.umem = false;
  bool sceneFileSpecified = false;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-b") {
      if (i + 2 < argc) {
        args.blockX = std::stoi(argv[i + 1]);
        args.blockY = std::stoi(argv[i + 2]);
        i += 2;
      } else {
        std::cerr << "Error: -b flag requires two arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " [options] -b <blockX> <blockY>"
                  << std::endl;
        return false;
      }
    } else if (std::string(argv[i]) == "-d") {
      if (i + 2 < argc) {
        args.width = std::stoi(argv[i + 1]);
        args.height = std::stoi(argv[i + 2]);
        i += 2;
      } else {
        std::cerr << "Error: -d flag requires two arguments to set output "
                     "image size.";
        std::cerr << "Usage: " << argv[0] << " [options] -d <width> <height>";
        return false;
      }
    } else if (std::string(argv[i]) == "-a") {
      if (i + 1 < argc) {
        args.umem = true;
      } else {
        std::cerr << "Error: -u flag requires one argument to run the unified "
                     "memory kernel.";
        std::cerr << "Usage: " << argv[0] << "[options] -u <sceneFile>";
      }
    } else {
      args.sceneFile = argv[i];
      sceneFileSpecified = true;
    }
  }

  if (!sceneFileSpecified) {
    std::cerr << "Error: No scene file specified." << std::endl;
    return false;
  }

  // Read the sphere count from the scene file
  std::ifstream sceneFile(args.sceneFile);
  if (!sceneFile.is_open()) {
    std::cerr << "Error: Could not open scene file " << args.sceneFile
              << std::endl;
    return false;
  }

  sceneFile >> args.sphereCount;
  if (sceneFile.fail()) {
    std::cerr << "Error: Could not read sphere count from scene file "
              << args.sceneFile << std::endl;
    return false;
  }

  return true;
}

std::string generateOutputFilename(const std::string &sceneFile,
                                   const std::string &prefix) {
  size_t startPos = sceneFile.find_last_of('/');
  if (startPos == std::string::npos) {
    startPos = 0;
  } else {
    startPos++;
  }
  size_t endPos = sceneFile.find_last_of('.');
  std::string baseName = sceneFile.substr(startPos, endPos - startPos);
  return prefix + baseName + ".ppm";
}

void syncRaytracer(Arguments args) {

  size_t numPixels = args.width * args.height;
  size_t framebufferSize = numPixels * sizeof(Vec3);

  // Allocate host memory
  Vec3 *h_framebuffer = (Vec3 *)malloc(framebufferSize);

  // Allocate device memory
  Vec3 *d_framebuffer;
  CUDA_CHECK(cudaMalloc(&d_framebuffer, framebufferSize));

  // Load scene from file
  std::vector<Sphere> spheres;
  loadScene(spheres, args.sceneFile);

  // Copy scene to device memory
  Sphere *d_spheres;
  CUDA_CHECK(cudaMalloc(&d_spheres, spheres.size() * sizeof(Sphere)));
  CUDA_CHECK(cudaMemcpy(d_spheres, spheres.data(),
                        spheres.size() * sizeof(Sphere),
                        cudaMemcpyHostToDevice));

  // Camera settings
  Vec3 camera_origin(0, 0, 0);
  float viewport_height = 2.0f;
  float viewport_width =
      (float)args.width / (float)args.height * viewport_height;
  float focal_length = 1.0f;

  dim3 blockSize(args.blockX, args.blockY);
  dim3 gridSize((args.width + blockSize.x - 1) / blockSize.x,
                (args.height + blockSize.y - 1) / blockSize.y);

  for (int i = 0; i < 10; i++) {

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    renderKernel<<<gridSize, blockSize>>>(
        d_framebuffer, args.width, args.height, camera_origin, viewport_width,
        viewport_height, focal_length, d_spheres, spheres.size());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_framebuffer, d_framebuffer, framebufferSize,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "GPU sync Rendering Time: " << elapsedTime / 1000.0f << " s\n";

    writeTrialToCsv(elapsedTime / 1000.0f, args.width, args.height,
                    args.sceneFile, "cuda_sync", args.sphereCount);
  }
  std::string outputFile =
      generateOutputFilename(args.sceneFile, "output_cuda_");

  savePPM(h_framebuffer, args.width, args.height, outputFile);

  CUDA_CHECK(cudaFree(d_framebuffer));
  CUDA_CHECK(cudaFree(d_spheres));
  CUDA_CHECK(cudaDeviceReset());
  free(h_framebuffer);
}

void unifiedMemoryRaytracer(Arguments args) {
  size_t numPixels = args.width * args.height;
  size_t framebufferSize = numPixels * sizeof(Vec3);

  // Allocate unified memory for framebuffer and spheres
  Vec3 *framebuffer;
  Sphere *spheres;

  CUDA_CHECK(cudaMallocManaged(&framebuffer, framebufferSize));

  // Load scene from file
  std::vector<Sphere> hostSpheres;
  loadScene(hostSpheres, args.sceneFile);
  CUDA_CHECK(cudaMallocManaged(&spheres, hostSpheres.size() * sizeof(Sphere)));

  // Copy scene data to unified memory
  memcpy(spheres, hostSpheres.data(), hostSpheres.size() * sizeof(Sphere));

  // Camera settings
  Vec3 camera_origin(0, 0, 0);
  float viewport_height = 2.0f;
  float viewport_width =
      (float)args.width / (float)args.height * viewport_height;
  float focal_length = 1.0f;

  dim3 blockSize(args.blockX, args.blockY);
  dim3 gridSize((args.width + blockSize.x - 1) / blockSize.x,
                (args.height + blockSize.y - 1) / blockSize.y);

  for (int i = 0; i < 10; i++) {

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    renderKernel<<<gridSize, blockSize>>>(
        framebuffer, args.width, args.height, camera_origin, viewport_width,
        viewport_height, focal_length, spheres, hostSpheres.size());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "GPU Unified Memory Rendering Time: " << elapsedTime / 1000.0f
              << " s\n";

    writeTrialToCsv(elapsedTime / 1000.0f, args.width, args.height,
                    args.sceneFile, "cuda_unified", args.sphereCount);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  std::string outputFile =
      generateOutputFilename(args.sceneFile, "output_cuda_unified_");

  savePPM(framebuffer, args.width, args.height, outputFile);

  CUDA_CHECK(cudaFree(framebuffer));
  CUDA_CHECK(cudaFree(spheres));
  CUDA_CHECK(cudaDeviceReset());
}

int main(int argc, char **argv) {
  Arguments args;
  if (!parseArguments(argc, argv, args)) {
    return 1;
  }

  if (args.umem) {
    unifiedMemoryRaytracer(args);
  } else {
    syncRaytracer(args);
  }

  return 0;
}
