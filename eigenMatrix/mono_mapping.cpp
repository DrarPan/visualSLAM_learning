#include <iostream>
#include <vector>
#include <fstream>
using namespace std;
#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.h>
using Sophus::SE3;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

// parameters
const int boarder = 20; // 边缘宽度
const int width = 640; // 宽度
const int height = 480; // 高度
const double fx = 481.2f; // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2; // NCC 取的窗口半宽度
const int ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1); // 窗口面积NCC
const double min_cov = 0.1; // 收敛判定:最小方差
const double max_cov = 10; // 发散判定:最大方差


bool readDatasetFiles(const string& path, vector<string>& color_image_files, vector<SE3>& poses);

bool update(const Mat& ref, const Mat& curr, const SE3& T_C_R, Mat& depth, Mat& depth_cov);

bool epipolarSearch(const Mat& ref, const Mat& curr, const SE3& T_C_R, const Vector2d& pt_ref,const double& depth_mu,const double& depth_cov,Vector2d& pt_curr);

bool updateDepthFilter(const Vector2d& pt_ref, const Vector2d& pt_curr, const SE3& T_C_R, Mat& depth, Mat& depth_cov);

double NCC( const Mat& ref, const Mat& curr, const Vector2d& pt_ref, const Vector2d& pt_curr );

inline double getBilinearInterpolatedValue( const Mat& img, const Vector2d& pt ) {
    uchar* d = & img.data[ int(pt(1,0))*img.step+int(pt(0,0)) ];
    double xx = pt(0,0) - floor(pt(0,0));
    double yy = pt(1,0) - floor(pt(1,0));
    return (( 1-xx ) * ( 1-yy ) * double(d[0]) +
        xx* ( 1-yy ) * double(d[1]) +
        ( 1-xx ) *yy* double(d[img.step]) +
        xx*yy*double(d[img.step+1]))/255.0;
}

bool plotDepth( const Mat& depth );

inline Vector3d px2cam ( const Vector2d px ) {
    return Vector3d ((px(0,0) - cx)/fx, (px(1,0) - cy)/fy, 1);
}

inline Vector2d cam2px ( const Vector3d p_cam ) {
    return Vector2d (p_cam(0,0)*fx/p_cam(2,0) + cx, p_cam(1,0)*fy/p_cam(2,0) + cy);
}

inline bool inside( const Vector2d& pt ) {
    return pt(0,0) >= boarder && pt(1,0)>=boarder && pt(0,0)+boarder<width && pt(1,0)+boarder<=height;
}

int main(int argc, char** argv){

}

bool update(const Mat& ref, const Mat& curr, const SE3& T_C_R, Mat & depth, Mat& depth_cov){
#pragma omp parallel for
   for(int x=boarder; x<width-boarder; x++)
#pragma omp parallel for
       for(int y=boarder; y<height-boarder; y++){
           if(depth_cov.ptr<double>(y)[x]<min_cov || depth_cov.ptr<double>(y)[x]>max_cov)
               continue;

           Vector2d pt_curr;
           bool ret = epipolarSearch(ref, curr, T_C_R, Vector2d(x,y), depth.ptr<double>(y)[x], sqrt(depth_cov.ptr<double>(y)[x]),pt_curr);

           updateDepthFilter(Vector2d(x,y),pt_curr,T_C_R,depth,depth_cov);
       }
}

bool epipolarSearch(const Mat& ref, const Mat& curr, const SE3& T_C_R, const Vector2d & pt_ref,
                    const double& depth_mu, const double & depth_cov, Vector2d& pt_curr){

    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref*depth_mu; //P vector of reference frame

    double d_min= depth_mu -3*depth_cov, d_max = depth_mu +3*depth_cov;

    if(d_min<0.1) d_min=0.1;

    //seems corresponding to curr frame (T_C_R: transformation from reference frame to current frame)
    Vector2d px_mean_curr = cam2px(T_C_R*P_ref); //depended on the mean of depth
    Vector2d px_max_curr = cam2px(T_C_R*(f_ref*d_max));
    Vector2d px_min_curr = cam2px(T_C_R*(f_ref*d_min));

    Vector2d epipolar_line = px_max_curr -px_min_curr;//line of epipolar_line

    Vector2d epipolar_direction = epipolar_line;
    epipolar_direction.normalize();

    double half_length =0.5*epipolar_line.norm();
    if(half_length>100) half_length =100;

    double best_ncc = -1.0;
    Vector2d best_px_curr;

    for(double l= -half_length; l<=half_length; l+=0.7){
        Vector2d px_curr = px_mean_curr + l*epipolar_direction;

        if(!inside(px_curr))
            continue;

        double ncc= NCC(ref,curr,pt_ref,px_curr);
        if(ncc>best_ncc){
            best_ncc=ncc;
            best_px_curr=px_curr;
        }
    }
    if(best_ncc <0.8)
        return false;

    pt_curr=best_px_curr;
    return true;
}

double NCC (const Mat& ref, const Mat& curr, const Vector2d& pt_ref, const Vector2d& pt_curr){
    double mean_ref = 0, mean_curr=0;
    vector<double> values_ref, values_curr;
    for ( int x=-ncc_window_size; x<=ncc_window_size; x++ )
        for ( int y=-ncc_window_size; y<=ncc_window_size; y++ ){
            double value_ref =double(ref.ptr<uchar>(int(y+pt_ref(1,0)))[int(x+pt_ref(0,0))])/255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr+ Vector2d(x,y));
            mean_curr += value_ref;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    double numerator =0, demoniator1=0, demoniator2 =0;

    for(int i=0; i<values_ref.size(); i++){
        numerator += (values_ref[i] -mean_ref)*(values_curr[i]-mean_curr);
        demoniator1 += (values_ref[i]- mean_ref) * (values_curr[i] - mean_curr);
        demoniator2 += (values_curr[i] -mean_curr) *(values_curr[i] -mean_curr);
    }
    return numerator/sqrt(demoniator1*demoniator2+1e-10);
}

bool updateDepthFilter(const Vector2d & pt_ref, const Vector2d& pt_curr, const SE& T_C_R, Mat& depth, Mat& depth_cov){
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.rotation_matrix() * f_curr;
    Vector2d b = Vector2d ( t.dot ( f_ref ), t.dot ( f2 ) );
    double A[4];
    A[0] = f_ref.dot ( f_ref );
    A[2] = f_ref.dot ( f2 );
    A[1] = -A[2];
    A[3] = - f2.dot ( f2 );
    double d = A[0]*A[3]-A[1]*A[2];
    Vector2d lambdavec =
    Vector2d ( A[3] * b ( 0,0 ) - A[1] * b ( 1,0 ),
    -A[2] * b ( 0,0 ) + A[0] * b ( 1,0 )) /d;
    Vector3d xm = lambdavec ( 0,0 ) * f_ref;
    Vector3d xn = t + lambdavec ( 1,0 ) * f2;
    Vector3d d_esti = ( xm+xn ) / 2.0;
    double depth_estimation = d_esti.norm();

    Vector3d p = f_ref*depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos( f_ref.dot(t)/t_norm );
    double beta = acos( -a.dot(t)/(a_norm*t_norm));
    double beta_prime = beta + atan(1/fx);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov*d_cov;

    double mu = depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    double sigma2 = depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];

    double mu_fuse = depth_cov.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))];
    double sigma_fuse2 = (sigma2* d_cov2) /(sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0))];

    depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = mu_fuse;
    depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = sigma_fuse2;

    return true;
}


