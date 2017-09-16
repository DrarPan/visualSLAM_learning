#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

#define TRAIN 0
#define LOADANDTEST 1

using namespace cv;
using namespace std;


int main( int argc, char** argv ){
#if(TRAIN)
    vector<Mat> images;
    for(int i=1;i<=10;i++){
        string path = "/home/hitrobot/test_ws/src/slambook/ch12/data/"+to_string(i)+".png";
        images.push_back(imread(path));
    }

    Ptr<Feature2D> detector = ORB::create();

    vector<Mat> descriptors;

    for(Mat& image:images){
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image,Mat(),keypoints,descriptor);
        descriptors.push_back(descriptor);
        //descriptorsize+=descriptor.size()[0];
        cout<<descriptor.size()<<endl;
    }

    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;

    vocab.create(descriptors);
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save("/home/hitrobot/test_ws/src/slambook/ch12/my_t_vocab.yml.gz",true);
#endif

#if(LOADANDTEST)
    cout<<"Starting..."<<endl;
    DBoW3::Vocabulary vocab("/home/hitrobot/test_ws/src/slambook/ch12/my_t_vocab.yml.gz");

    if(vocab.empty()){
        cout<<"Vocabulary does not exist."<<endl;
        return -1;
    }

    vector<Mat> images;
    for(int i=1;i<=10;i++){
        string path = "/home/hitrobot/test_ws/src/slambook/ch12/data/"+to_string(i)+".png";
        images.push_back(imread(path));
    }

    Ptr<Feature2D> detector=ORB::create();
    vector<Mat> descriptors;

    for ( Mat& image:images ){
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }

    cout<<"comparing images with images"<<endl;

    for(int i=0;i<images.size();i++){
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i],v1);

        for(int j=i; j<images.size();j++){
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j],v2);
            double score = vocab.score(v1,v2);
            cout<<"image "<<i<<" vs image "<<j<<" : "<<score<<endl;
        }
        cout<<endl;
    }

    cout<<"comparing images with database "<<endl;

    DBoW3::Database db(vocab,false,0);

    for(int i=0; i<descriptors.size();i++)
        db.add(descriptors[i]);
    cout<<"database info: "<<db<<endl;

    for(int i=0;i<descriptors.size();i++){
        DBoW3::QueryResults ret;
        db.query(descriptors[i],ret,4);
        cout<<"searching for image"<<i<<" returns "<<ret<<endl<<endl;
    }
#endif
    cout<<"done."<<endl;
}
