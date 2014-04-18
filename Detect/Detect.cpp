#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: Detect image_file" << endl;
        return -1;
    }

    Mat img;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat all_descriptors;
    Ptr<FeatureDetector> detector(new SurfFeatureDetector(100));
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
    Mat vocabulary;
    FileStorage fs;

    if (fs.open("vocabulary.yml", FileStorage::READ))
    {
        cout << "Reading vocabulary from file..." << endl;
        fs["vocabulary"] >> vocabulary;
        fs.release();
    }

}