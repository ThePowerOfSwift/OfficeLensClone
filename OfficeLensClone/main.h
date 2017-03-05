#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2\core\mat.hpp>

// Most images are too big for 1080p screens, scale them down by this factor
#define IMG_SCALE_FACTOR 4
#define WEBCAM 1
#define LINE_THICKNESS 10

// Runs auto_crop constantly with input from the webcam.
void office_lens();
// Automatically crops out an object in an image.  Returns 4 ints which represent the top left and bottom right point.
std::array<int, 4> auto_crop(cv::Mat &image);
// Automatically calls test_auto_crop on every image in the data folder.
void auto_test();
// Returns the percentage of overlap from the output of auto_crop compared to the true boundary.
double test_auto_crop(const std::string &path, const cv::Rect &true_boundary, bool display_results);
// Reads in the true boundaries for use in auto_test
std::vector<std::array<int, 4>> get_true_boundaries(const std::string &path);

double get_rectangle_overlap(const cv::Rect &lhs, const cv::Rect &rhs);