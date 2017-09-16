#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc,char** argv){
    Mat img_1=imread("/home/hitrobot/pll_ws/slam_learning_gx/build-eigenMatrix-Desktop_Qt_5_7_0_GCC_64bit-Default/left.png",CV_LOAD_IMAGE_COLOR);
    Mat img_2=imread("/home/hitrobot/pll_ws/slam_learning_gx/build-eigenMatrix-Desktop_Qt_5_7_0_GCC_64bit-Default/right.png",CV_LOAD_IMAGE_COLOR);

    ORB orb;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    orb(img_1,Mat(),keypoints_1,descriptors_1);
    orb(img_2,Mat(),keypoints_2,descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("Feature ORB",outimg1);
    cvWaitKey(1);
    drawKeypoints(img_1,keypoints_1,outimg1,DrawMatchesFlags::DEFAULT);

    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1,descriptors_2,matches);

    double min_dist=10000, max_dist=0;

    for(int i=0; i <descriptors_1.rows;i++){

    }

}
