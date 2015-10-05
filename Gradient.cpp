//============================================================================
// Name        : Gradient.cpp
// Author      : Dailos Guerra
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <cv.h>
#include <highgui.h>

#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

void zeroCrossings(IplImage *src, IplImage *zeroCrossImg, double threshold);
#define PI 3.14159265

int main(int argc, char** argv)
{
  IplImage* img = cvLoadImage( "../Ejercicio4_1.pgm" );

  //Create gray-scale version of the original image
  IplImage* src = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
  cvCvtColor( img, src, CV_BGR2GRAY );

  //Create destination image with same type and size
  IplImage* dst = cvCloneImage(src);
  dst->origin = src->origin;

  //Create transformation (rotation) matrix
  CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
  CvPoint2D32f center = cvPoint2D32f(src->width/2, src->height/2);
  double angle = atan (float(src->height)/float(src->width)) * 180 / PI;
  int diagonal_size = sqrt(pow(src->height,2)+pow(src->width,2));
  double scale = (float)src->width/(float)diagonal_size;
  cv2DRotationMatrix(center, angle , scale, rot_mat);

  //Apply rotation to the image
  cvWarpAffine(src, dst, rot_mat);

  //Extract central row, used to be the diagonal
  CvMat* row = cvCreateMat(1, dst->width, CV_32FC1);
  cvGetRow(dst, row, dst->height/2);
  double max_value = 0.0, min_value = 0.0;
  cvMinMaxLoc( row, &min_value, &max_value);

  //Create image to plot pixels intensities
  IplImage *row_plot = cvCreateImage(cvSize(300,240), 8, 1);
  cvSet( row_plot, cvScalarAll(255), 0 );
  float w_scale = ((float)row_plot->width)/dst->width;

  // plot diagonal intensities
  for( int i = 0; i < dst->width; i++ )
  {
    cvRectangle( row_plot, cvPoint((int)i*w_scale , row_plot->height),
                 cvPoint((int)(i+1)*w_scale, row_plot->height - cvRound(cvGetReal1D(row,i))),
                  cvScalar(0), -1, 8, 0 );
  }

  //Vertical Sobel Gradient
  IplImage *v_Sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  float v_Sobel_kernel[] = { -1, -2, -1,
                              0,  0,  0,
                              1,  2,  1 };
  CvMat v_Sobel;
  cvInitMatHeader(&v_Sobel, 3, 3, CV_32FC1, v_Sobel_kernel);
  cvFilter2D(src, v_Sobel_img, &v_Sobel);

  //Horizontal Sobel Gradient
  IplImage *h_Sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  float h_Sobel_kernel[] = { -1, 0,  1,
                             -2, 0,  2,
                             -1, 0,  1 };
  CvMat h_Sobel;
  cvInitMatHeader(&h_Sobel, 3, 3, CV_32FC1, h_Sobel_kernel);
  cvFilter2D(src, h_Sobel_img, &h_Sobel);

  IplImage *mod_Sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  IplImage *contours_Sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  IplImage *modH_Sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  IplImage *modV_Sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );

  //Summation of the modules of vertical and horizontal Sobel components
  cvAbs(h_Sobel_img, modH_Sobel_img);
  cvAbs(v_Sobel_img, modV_Sobel_img);
  cvAdd(modV_Sobel_img, modH_Sobel_img, mod_Sobel_img);

  //Apply threshold to get the contours
  cvThreshold( mod_Sobel_img, contours_Sobel_img , 100, 0xff, CV_THRESH_BINARY );

  //Apply Sobel operator to the image explicitly
  IplImage* sobel_tmp = cvCreateImage( cvGetSize(src), IPL_DEPTH_16S, 1);
  IplImage *sobel_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  IplImage *sobel_contours_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  cvSobel(src, sobel_tmp, 1, 1, 3);
  cvConvertScale(sobel_tmp,sobel_img);
  cvThreshold( sobel_img, sobel_contours_img , 0.01, 0xff, CV_THRESH_BINARY );

  //Apply Laplace operator to the image explicitly
  IplImage* laplace_tmp = cvCreateImage( cvGetSize(src), IPL_DEPTH_16S, 1);
  IplImage *contours = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );

  cvLaplace(src, laplace_tmp, 3);
  zeroCrossings(laplace_tmp, contours, 40);

  namedWindow("Src",1);
  imshow( "Src", row_plot );
  namedWindow("Row Plot",1);
  imshow( "Row Plot", row_plot );
  namedWindow("Vertical Sobel",1);
  imshow( "Vertical Sobel", v_Sobel_img );
  namedWindow("Horizntal Sobel",1);
  imshow( "Horizntal Sobel", h_Sobel_img );
  namedWindow("Module Sobel",1);
  imshow( "Module Sobel", mod_Sobel_img );
  namedWindow("Contours",1);
  imshow( "Contours", contours_Sobel_img );
  namedWindow("Sobel",1);
  imshow( "Sobel", sobel_img );
  namedWindow("Sobel Contours",1);
  imshow( "Sobel Contours", sobel_contours_img );
  namedWindow( "Contours", 1 );
  imshow( "Contours", contours );

  waitKey();

/*
  cvNamedWindow("Original",1);
  cvShowImage( "Original", src );
  cvNamedWindow("Laplace Contours",1);
  cvShowImage( "Laplace Contours", contours );
*/
}

void zeroCrossings(IplImage *src, IplImage *zeroCrossImg, double threshold)
{
  IplImage* laplace_mask = cvCreateImage( cvGetSize(src), IPL_DEPTH_16S, 1);
  IplImage* laplace_zerocross = cvCreateImage( cvGetSize(src), IPL_DEPTH_16S, 1);
  IplImage *laplace_contours_img = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  IplImage *laplace_tmp_contours = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
  IplImage *threshold_mask = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );

  //create kernel to detect horizontal zerocrossing
  float h[] = { 0,  0, 0,
	            -1, 0, 1,
                0,  0, 0 };
  CvMat h_kernel;
  cvInitMatHeader(&h_kernel, 3, 3, CV_32FC1, h);
  cvFilter2D(src, laplace_zerocross, &h_kernel);
  cvAbs(laplace_zerocross, laplace_mask);
  cvConvertScale(laplace_mask, threshold_mask, 1, 0);
  cvCmpS(threshold_mask, threshold, threshold_mask, CV_CMP_GT);
  cvConvertScale(src, laplace_tmp_contours, 1, 0);
  cvThreshold(laplace_tmp_contours, laplace_tmp_contours, 0, 1, CV_THRESH_BINARY);
  cvFilter2D(laplace_tmp_contours, laplace_contours_img, &h_kernel);
  cvAnd(laplace_contours_img, threshold_mask, zeroCrossImg);

  //kernel for vertical zerocrossing
  float v[] = { 0,  1, 0,
	            0, 0, 0,
                0,  -1, 0 };
  CvMat v_kernel;
  cvInitMatHeader(&v_kernel, 3, 3, CV_32FC1, v);
  cvFilter2D(src, laplace_zerocross, &v_kernel);
  cvConvertScale(src, laplace_tmp_contours, 1, 0);
  cvThreshold(laplace_tmp_contours, laplace_tmp_contours, 0, 1, CV_THRESH_BINARY);
  cvFilter2D(laplace_tmp_contours, laplace_contours_img, &v_kernel);
  cvAnd(laplace_contours_img, threshold_mask, laplace_contours_img);
  cvAdd(zeroCrossImg, laplace_contours_img, zeroCrossImg);

  //kernel for right diagonal zerocrossing
  float rd[] = { 0,  0, 1,
	            0, 0, 0,
                -1,  0, 0 };
  CvMat rd_kernel;
  cvInitMatHeader(&rd_kernel, 3, 3, CV_32FC1, rd);
  cvFilter2D(src, laplace_zerocross, &rd_kernel);
  cvConvertScale(src, laplace_tmp_contours, 1, 0);
  cvThreshold(laplace_tmp_contours, laplace_tmp_contours, 0, 1, CV_THRESH_BINARY);
  cvFilter2D(laplace_tmp_contours, laplace_contours_img, &rd_kernel);
  cvAnd(laplace_contours_img, threshold_mask, laplace_contours_img);
  cvAdd(zeroCrossImg, laplace_contours_img, zeroCrossImg);

  //kernel for left diagonal zerocrossing
  float ld[] = { 1,  0, 0,
	            0, 0, 0,
                0,  0, -1 };
  CvMat ld_kernel;
  cvInitMatHeader(&ld_kernel, 3, 3, CV_32FC1, ld);
  cvFilter2D(src, laplace_zerocross, &ld_kernel);
  cvConvertScale(src, laplace_tmp_contours, 1, 0);
  cvThreshold(laplace_tmp_contours, laplace_tmp_contours, 0, 1, CV_THRESH_BINARY);
  cvFilter2D(laplace_tmp_contours, laplace_contours_img, &ld_kernel);
  cvAnd(laplace_contours_img, threshold_mask, laplace_contours_img);
  cvAdd(zeroCrossImg, laplace_contours_img, zeroCrossImg);

  cvThreshold(zeroCrossImg, zeroCrossImg, 0, 0xff, CV_THRESH_BINARY);
}
