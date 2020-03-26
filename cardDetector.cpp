#include <iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void sharpenImage(Mat& grayScaleImage)
{
	Mat gaussianFilteredImage;
	Mat unsharpMask;
	Mat outputImage;

	GaussianBlur(grayScaleImage, unsharpMask, Size(25, 25), 30, 30);

	addWeighted(grayScaleImage, 2.4, unsharpMask, -1, -15, outputImage);

	grayScaleImage = outputImage;
	
	imshow("Unblurred", outputImage);
}

void removeGradient(Mat& grayScaleImage, Mat& outputImage)
{
	//checking all the possible errors
	if (grayScaleImage.empty() || grayScaleImage.channels() != 1 || grayScaleImage.type() != CV_8UC1)
	{
		cout << "Wrong input!";
		return;
	}

	outputImage = Mat(grayScaleImage.size(), CV_8UC1);
	
	//initializing minimum and maximum values
	int min = grayScaleImage.at<uchar>(0, 0);
	int max = grayScaleImage.at<uchar>(0, 0);

	//finding minimum and maximum value by checking every single pixel in grayScaleImage
	for (int i = 0; i < grayScaleImage.rows; i++)
	{
		for (int j = 0; j < grayScaleImage.cols; j++)
		{
			if (grayScaleImage.at<uchar>(i, j) < min)
			{
				min = grayScaleImage.at<uchar>(i, j);
			}

			if (grayScaleImage.at<uchar>(i, j) > max)
			{
				max = grayScaleImage.at<uchar>(i, j);
			}
		}
	}

	//stretching histogram
	for (int i = 0; i < outputImage.rows; i++)
	{
		for (int j = 0; j < outputImage.cols; j++)
		{
			outputImage.at<uchar>(i, j) = 255 * (grayScaleImage.at<uchar>(i, j) - min) / (max - min);
			//cout << out.at<uchar>(i, j);
		}
	}
	//resize(outputImage.clone(), outputImage, Size(), 0.7, 0.7);
	imshow("Stretched", outputImage);
}

void contoursFinder(Mat& grayScaleImage, Mat& cannyImage)
{	
	//creating a vector in which we can store contours
	vector<vector<Point>> contours;
	//vector for hierarchy of the specific contours
	vector<Vec4i> hierarchy;
	RNG rng(1);
	//Canny function returns the image with thin contours but we need to eliminate all the interior contours as well
	Canny(grayScaleImage, cannyImage, 0, 255, 3);
	//resize(cannyImage.clone(), cannyImage, Size(), 0.7, 0.7);
	imshow("cannyImage", cannyImage);

	//morphological operation needed to be done to make findContours able to find any contours
	dilate(cannyImage, grayScaleImage, Mat());
	imshow("dilatedImage", grayScaleImage);

	//black image made from the grayScaleImage where the specific contours will be sent
	cannyImage = Mat::zeros(grayScaleImage.size(), CV_8UC1);
	Mat colouredImage = Mat::zeros(cannyImage.size(), CV_8UC3);
	//stackOverflow
	findContours(grayScaleImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point(0, 0));

	//hierarchy will be changing after every iteration because every card has the next contour at the same level until it reaches the final card
	for (int i = 0; i >= 0; i = hierarchy[i][0])
	{
		//cout << hierarchy[i][0] << endl;
		//calculating the size of the blob so we can ignore the small symbols and numbers on a card to focus on a number of main symbols
		if (contourArea(contours[i], false) / arcLength(contours[i], true) > 6.9)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			//segmentation of contours with the specfic size
			drawContours(colouredImage, contours, i, color, 1, LINE_4, hierarchy, 0);
			drawContours(cannyImage, contours, i, Scalar(255, 255, 255), 1, LINE_4, hierarchy, 0);
		}
	}
	imshow("Test", cannyImage);
	imshow("Segmented Image", colouredImage);
}


int* countSymbols(Mat& cannyImage, int sumTab[4])
{
	int cardID = 0;
	//creating a vector in which we can store contours
	vector<vector<Point>> contours;
	//vector for hierarchy of the specific contours
	vector<Vec4i> hierarchy;
	int contourID[4];

	//stackOverflow
	findContours(cannyImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
	resize(cannyImage.clone(), cannyImage, Size(), 0.7, 0.7);
	imshow("Final image", cannyImage);

	for (int i = 0; i < hierarchy.size(); i++)
	{
		//if we find a contour that has no parent means we are inside each card
		if (hierarchy[i][3] == -1)
		{
			//finding all the children of each card - hierarchy[i][2] is an internal contour of a card
			contourID[cardID] = hierarchy[i][2];
			cardID++;
		}
	}

	for (int cardID = 0; cardID < 4; cardID++)
	{
		sumTab[cardID] = 0;
		for (int i = 0; i < hierarchy.size(); i++)
		{
			//looking for all the elements which parent is the internal contour of a card
			if (hierarchy[i][3] == contourID[cardID])
			{
				sumTab[cardID]++;
			}
		}
	}

	for (int cardID = 0; cardID < 4; cardID++)
	{
		cout << "Card nr " << (cardID + 1) << ": " << sumTab[cardID]<< endl;;
	}

	return sumTab;
}

void countVariation(Mat& cannyImage, int sumTab[4])
{
	int sum = 0;

	for (int i = 0; i < 4; i++)
	{
		sum += sumTab[i];
	}
	double tempTab[4];
	double tempTabSum = 0;

	cout << "Sum of the cards: " << sum << endl;
	double mean = double(sum) / 4;
	cout << "Mean value: " << mean << endl;

	for (int i = 0; i < 4; i++)
	{
		tempTab[i] = (sumTab[i] - mean) * (sumTab[i] - mean);
		tempTabSum += tempTab[i];
	}

	cout << "Temporary sum of the squares of the value and mean difference: " << tempTabSum << endl;
	cout << "Variance value: " << tempTabSum / 4;
}

void userDecision(Mat &srcImage, Mat& grayScaleImage, Mat& laplacianImage, Mat& cannyImage)
{
	cout << "Welcome to the cardDetector programme! Choose the type of the image which you would like to process: " << endl;
	cout << "1. Original image" << endl;
	cout << "2. Blurred image" << endl;
	cout << "3. Gradient image" << endl;
	cout << "4. Salt & Pepper image" << endl;

	Mat outputImage;

	int sumTab[4];
	
	int userD;

	cin >> userD;

	switch (userD)
	{

	case 1:
		cout << "You decided to pick the original image. " << endl;

		srcImage = imread("12390186-2019-11-21-144455.tif", 1);

		if (srcImage.empty())
			return;

		namedWindow("Original image", WINDOW_AUTOSIZE);
		imshow("Original image", srcImage);
		cvtColor(srcImage, grayScaleImage, COLOR_BGR2GRAY);
		//resize(grayScaleImage.clone(), grayScaleImage, Size(), 0.7, 0.7);
		namedWindow("Grayscaled image", WINDOW_AUTOSIZE);
		imshow("Grayscaled image", grayScaleImage);
		medianBlur(grayScaleImage, grayScaleImage, 5);
		//resize(grayScaleImage.clone(), grayScaleImage, Size(), 0.7, 0.7);
		namedWindow("Filtered Image", WINDOW_AUTOSIZE);
		imshow("Filtered Image", grayScaleImage);
		threshold(grayScaleImage.clone(), grayScaleImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
		namedWindow("Binarized Image", WINDOW_AUTOSIZE);
		imshow("Binarized Image", grayScaleImage);
		contoursFinder(grayScaleImage, cannyImage);
		countSymbols(cannyImage, sumTab);
		countVariation(cannyImage, sumTab);
		waitKey(0);

		break;

	case 2:
		cout << "You decided to pick the blurred image. " << endl;

		srcImage = imread("12390186-2019-11-21-144455_blur.tif", 1);

		if (srcImage.empty())
			return;
		
		namedWindow("Blurred image", WINDOW_AUTOSIZE);
		imshow("Blurred image", srcImage);
		cvtColor(srcImage, grayScaleImage, COLOR_BGR2GRAY);
		namedWindow("Grayscaled image", WINDOW_AUTOSIZE);
		imshow("Grayscaled image", grayScaleImage);

		for (int i = 0; i < 2; i ++)
		{
			sharpenImage(grayScaleImage);
		}
		
		

		namedWindow("Fixed image", WINDOW_AUTOSIZE);
		imshow("Fixed image", grayScaleImage);
		medianBlur(grayScaleImage.clone(), grayScaleImage, 5);
		threshold(grayScaleImage.clone(), grayScaleImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
		namedWindow("Binarized Image", WINDOW_AUTOSIZE);
		imshow("Binarized Image", grayScaleImage);
		contoursFinder(grayScaleImage, cannyImage);		
		countSymbols(cannyImage, sumTab);
		countVariation(cannyImage, sumTab);

		waitKey(0);		

		break;

	case 3:
		cout << "You decided to pick the gradient image. " << endl;

		srcImage = imread("12390186-2019-11-21-144455_gradient.tif", 1);

		if (srcImage.empty())
			return;
		
		namedWindow("Gradient image", WINDOW_AUTOSIZE);
		imshow("Gradient image", srcImage);
		cvtColor(srcImage, grayScaleImage, COLOR_BGR2GRAY);
		namedWindow("Grayscaled image", WINDOW_AUTOSIZE);
		imshow("Grayscaled image", grayScaleImage);
		removeGradient(grayScaleImage, outputImage);
		contoursFinder(outputImage, cannyImage);
		countSymbols(cannyImage, sumTab);
		countVariation(cannyImage, sumTab);

		waitKey(0);

		break;

	case 4:
		cout << "You decided to pick the salt & pepper image. " << endl;

		srcImage = imread("12390186-2019-11-21-144455_salt_pepper.tif", 1);

		if (srcImage.empty())
			return;

		namedWindow("Salt & Pepper image", WINDOW_AUTOSIZE);
		imshow("Salt & Pepper image", srcImage);
		cvtColor(srcImage, grayScaleImage, COLOR_BGR2GRAY);
		namedWindow("Grayscaled image", WINDOW_AUTOSIZE);
		imshow("Grayscaled image", grayScaleImage);
		medianBlur(grayScaleImage.clone(), grayScaleImage, 5);
		threshold(grayScaleImage.clone(), grayScaleImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
		namedWindow("Binarized Image", WINDOW_AUTOSIZE);
		imshow("Binarized Image", grayScaleImage);
		contoursFinder(grayScaleImage, cannyImage);
		countSymbols(cannyImage, sumTab);
		countVariation(cannyImage, sumTab);
		waitKey(0);

		break;

	default:
		cout << "There is no option with this number! Choose again." << endl;
		userDecision(srcImage, grayScaleImage, laplacianImage, cannyImage);
	}

}

int main()
{
	Mat srcImage;
	Mat grayScaleImage;
	Mat laplacianImage;
	Mat cannyImage;

	userDecision(srcImage, grayScaleImage, laplacianImage, cannyImage);
}

