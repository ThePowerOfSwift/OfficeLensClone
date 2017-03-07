#include <array>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\videoio.hpp>

#include "main.h"

int main()
{
	int choice = -1;

	do
	{
		std::cout << "1. Office Lens\n2. Test Office Lens" << std::endl;
		std::cin >> choice;
	} while (choice != 1 && choice != 2);

	if (choice == 1)
		office_lens();
	else
		auto_test();
}

// Runs auto_crop constantly with input from the webcam.
void office_lens()
{
	cv::Scalar color(0, 255, 0);
	cv::VideoCapture cap(WEBCAM);

	while (true)
	{
		cv::Mat frame;
		cap >> frame;

		cv::Mat grayscale;
		cv::cvtColor(frame, grayscale, CV_RGB2GRAY);

		std::array<int, 4> res = auto_crop(grayscale);

		cv::Mat frame_with_rect;
		frame.copyTo(frame_with_rect); // If we save the frame, we don't want the rectangle too.
		cv::rectangle(frame_with_rect, cv::Rect(cv::Point(res[0], res[1]), cv::Point(res[2], res[3])), color, LINE_THICKNESS);

		cv::imshow("OfficeLens", frame_with_rect);

		char key = cv::waitKey(1);

		if (key == 'q')
			break;
		else if (key == 's')
		{
			// res is originally in the format x1, y1, x2, y2, but we need x, y, width, height.  Convert it
			crop(frame, cv::Rect(res[0], res[1], res[2] - res[0], res[3] - res[1]));
			cv::imwrite(get_time_str() + ".png", frame);
		}
	}
}

/*
@brief Returns the top left point (sx, sy) and bottom right point (sx2, sy2) that define the object.
@param Mat : grayscale image
*/
std::array<int, 4> auto_crop(cv::Mat &image)
{
	int sx = 0, sy = 0, sx2 = 0, sy2 = 0;

	cv::threshold(image, image, 0, 255, cv::THRESH_OTSU);

	cv::Mat sum_row;
	cv::reduce(image, sum_row, 0, CV_REDUCE_SUM, CV_32S);
	cv::Mat sum_row_over_avg = sum_row > cv::mean(sum_row)[0];

	cv::Mat sum_col;
	cv::reduce(image, sum_col, 1, CV_REDUCE_SUM, CV_32S);
	cv::Mat sum_col_over_avg = sum_col > cv::mean(sum_col)[0];

	const auto begin_row = sum_row_over_avg.ptr<unsigned char>(0);
	const auto end_row = begin_row + sum_row_over_avg.cols;
	const auto row_iterator = std::find(begin_row, end_row, (unsigned char)255);

	sx = row_iterator == end_row ? 0 : std::distance(begin_row, row_iterator);

	const auto begin_col = sum_col_over_avg.ptr<unsigned char>(0);
	const auto end_col = begin_col + sum_col_over_avg.rows;
	const auto col_iterator = std::find(begin_col, end_col, (unsigned char)255);

	sy = col_iterator == end_col ? 0 : std::distance(begin_col, col_iterator);

	// There are no reverse iterators that I'm aware of, so we can't use find to find the values at the end.
	// OpenCV's Matrix needs a better iterator interface considering how many standard library functions use them.
	const auto col_ptr = sum_row_over_avg.ptr<unsigned char>(0);
	for (int col = sum_row_over_avg.cols - 1; col != -1; --col)
	{
		if ((int)col_ptr[col] == 255)
		{
			sx2 = col;
			break;
		}
	}

	for (int row = sum_col_over_avg.rows - 1; row != -1; --row)
	{
		const auto row_ptr = sum_col_over_avg.ptr<unsigned char>(row);
		if ((int)*row_ptr == 255)
		{
			sy2 = row;
			break;
		}
	}	

	return { sx, sy, sx2, sy2 };
}

void auto_test()
{
	const int img_amt = 25;
	const std::string main_path = "data\\input_";
	const std::string file_type = ".jpg";

	const std::vector<std::array<int, 4>> boundaries = get_true_boundaries("data\\truth.csv");

	int total_points = 0;
	double avg_overlap = 0.0;

	for (int i = 1; i <= img_amt; i++)
	{
		std::string path = main_path + std::to_string(i) + file_type;
		std::array<int, 4> pos = boundaries[i - 1];

		cv::Rect true_boundary(pos[0], pos[1], pos[2], pos[3]);
		double overlap = test_auto_crop(path, true_boundary, false);
		avg_overlap += overlap;

		if (overlap >= 0.8)
			total_points += 2;
		else if (overlap >= 0.5)
			total_points += 1;
	}

	avg_overlap /= img_amt;

	std::cout << "Total Points: " << total_points << " Avg Overlap: " << avg_overlap << std::endl;
}

double test_auto_crop(const std::string &path, const cv::Rect &true_boundary, bool display_results)
{
	cv::Mat image = cv::imread(path);

	if (image.empty())
	{
		std::cout << "Unable to load image at \"" << path << "\"." << std::endl;
		return 0.0;
	}

	cv::Mat grayscale;
	cv::cvtColor(image, grayscale, CV_RGB2GRAY);

	std::array<int, 4> res = auto_crop(grayscale);

	double overlap = get_rectangle_overlap(
		cv::Rect(cv::Point(res[0], res[1]), cv::Point(res[2], res[3])),
		true_boundary);

	std::cout << path << " overlap %: " << overlap << ".";

	if (display_results)
	{
		cv::rectangle(image, cv::Rect(cv::Point(res[0], res[1]), cv::Point(res[2], res[3])), cv::Scalar(0, 255, 0), LINE_THICKNESS);
		cv::resize(image, image, cv::Size(image.cols / IMG_SCALE_FACTOR, image.rows / IMG_SCALE_FACTOR));

		std::cout << " Press enter to continue.";

		cv::imshow("image", image);
		cv::waitKey(1);
		std::cin.ignore();
	}

	std::cout << std::endl;

	return overlap;
}

std::vector<std::array<int, 4>> get_true_boundaries(const std::string &path)
{
	std::vector<std::array<int, 4>> boundaries;
	std::ifstream input(path);

	std::string curr_line;
	while (std::getline(input, curr_line))
	{
		std::array<int, 4> rectangle;
		std::istringstream in(curr_line);
		
		int n, index = 0;
		while (in >> n)
		{
			rectangle[index++] = n;

			if (in.peek() == ',')
				in.ignore();
		}

		boundaries.push_back(rectangle);
	}

	return boundaries;
}

inline
void crop(cv::Mat &image, const cv::Rect &area)
{
	image = image(area);
}

// Thanks to http://stackoverflow.com/a/16358111
std::string get_time_str()
{
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y--%H-%M-%S");

	return oss.str();
}

double get_rectangle_overlap(const cv::Rect &lhs, const cv::Rect &rhs)
{
	cv::Rect intersect = lhs & rhs;
	return intersect.area() / (double)lhs.area();
}