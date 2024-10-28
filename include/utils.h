#ifndef UTILS_H
#define UTILS_H

#include "raytracer.h"
#include <string>
#include <vector>

bool savePPM(const Vec3 *framebuffer, int width, int height,
             const std::string &filename);

bool readPPM(const std::string &filename, std::vector<Vec3> &framebuffer,
             int &width, int &height);

bool compareImages(const std::vector<Vec3> &image1,
                   const std::vector<Vec3> &image2, int width, int height,
                   float tolerance, const std::string &outputFile,
                   int sphereCount);

void writeTrialToCsv(double trial_time, int width, int height,
                     const std::string &sceneFile, const std::string &type,
                     int numSpheres);
void loadScene(std::vector<Sphere> &spheres, const std::string &filename);

#endif // UTILS_H
