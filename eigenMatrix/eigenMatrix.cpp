#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "sophus/so3.h"
#include "sophus/se3.h"

int main(int argc, char** argv){
if(0){ //eigen
    Eigen::Matrix<float,2,3> matrix_23;
    Eigen::Vector3d v_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_dynamic;

    Eigen::MatrixXd matrix_x;

    matrix_23 << 1,2,3,4,5,6;
    cout<<matrix_23<<endl;

    for(int i=0;i<1;i++)
        for(int j=0;j<2;j++)
            cout<<matrix_23(i,j)<<endl;

    v_3d<< 3,2,1;

    Eigen::MatrixXd result = matrix_23.cast<double>()*v_3d;
    cout<<"Result: "<<result<<endl;

    matrix_33 = Eigen::Matrix3d::Random();

    cout<<matrix_33<<endl;

    cout<<matrix_33.transpose()<<endl;
    cout<<matrix_33.sum()<<endl;
    cout<<matrix_33.trace()<<endl;
    cout<<matrix_33.inverse()<<endl;
    cout<<matrix_33.determinant()<<endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33);

    cout<<"Eigen Value: "<<eigen_solver.eigenvalues()<<endl;
    cout<<"Eigen Vector:"<<eigen_solver.eigenvectors()<<endl;

    Eigen::Vector3d x;

    x=matrix_33.inverse()*v_3d;
    cout<<"x="<<x<<endl;

    auto xxx=matrix_33.colPivHouseholderQr().solve(v_3d);
    cout<<"x="<<xxx<<endl;
}
if(0){ //eigen about transform
    Eigen::Matrix3d rotation_matrix =Eigen::Matrix3d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI/2,Eigen::Vector3d(0,0,1));

    //cout<<"rotation matrix = \n"<<rotation_vector.matrix()<<endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    Eigen::Vector3d v(1,0,0);
    Eigen::Vector3d v_rotated = rotation_vector * v;

    cout<<"(1,0,0) after rotation = \n"<<v_rotated<<endl;

    Eigen::Vector3d euler_angles =rotation_matrix.eulerAngles(0,1,2);
    cout<<"roll pitch yaw = \n"<< euler_angles <<endl;

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1,0,0));

    cout<<"Transform matrix: \n"<<T.matrix()<<endl;

    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);

    cout<<"Quaternion = \n"<<q.coeffs()<<endl;

    v_rotated = q*v; //mathmatically qvq^{-1}

    cout<<"(1,0,0) after rotation = \n"<<v_rotated<<endl;

    return 0;
}

if(0){
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    //Sophus::SO3: Lie Group
    Sophus::SO3 SO3_R(R);
    Sophus::SO3 SO3_v( 0, 0, M_PI/2 );
    Eigen::Quaterniond q(R);
    Sophus::SO3 SO3_q( q );

    cout<<"SO(3) from matrix: "<<SO3_R<<endl;
    cout<<"SO(3) from vector: "<<SO3_v<<endl;
    cout<<"SO(3) from quaternion: "<<SO3_q<<endl;

    Eigen::Vector3d so3 = SO3_R.log();
    cout<<"so3 = "<<so3<<endl;

    cout<<"so3 hat="<<Sophus::SO3::hat(so3)<<endl;

    cout<<"so3 hat vee = "<<Sophus::SO3::vee(Sophus::SO3::hat(so3))<<endl;
    /////////////////////////////////////////////////////

    Eigen::Vector3d t(0.5,0,0);
    Sophus::SE3 SE3_Rt(R, t);
    Sophus::SE3 SE3_qt(q,t);
    cout<<"SE3 from R,t= "<<endl<<SE3_Rt<<endl;
    cout<<"SE3 from q,t="<<endl<<SE3_qt<<endl;
    Sophus::Vector6d update_se3;
    update_se3.setZero();
    update_se3(0,0)=1e-5;

    Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3)*SE3_Rt;
    cout<<"SE3 updated ="<<endl<<SE3_updated.matrix()<<endl;

    //in se3 space
    SE3_updated = Sophus::SE3::exp(update_se3+SE3_Rt.log());
    cout<<"SE3 updated2 ="<<endl<<SE3_updated.matrix()<<endl;
}



}

