#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree.hpp>

using namespace std;
using namespace cv;

int main()
{
    /* get filepaths vector */
    string filedir = "C:/Users/Meng/Downloads/ORL faces/";
    fstream filelist(filedir + "list");
    vector<string> filepaths;
    string filename;

    while (getline(filelist, filename))
        filepaths.push_back(filedir + filename);

    /* build BOW vocabulary */
    Mat img;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat all_descriptors;
    Ptr<FeatureDetector> detector(new SurfFeatureDetector(100));
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
    BOWKMeansTrainer bowtrainer(100);

    cout << "Extracting descriptors..." << endl;
    for (int i = 0; i < filepaths.size(); i++)
    {
        img = imread(filepaths[i], 0);
        detector->detect(img, keypoints);
        extractor->compute(img, keypoints, descriptors);
        all_descriptors.push_back(descriptors);
    }

    cout << "Building vocabulary..." << endl;
    Mat vocabulary = bowtrainer.cluster(all_descriptors);

    /* build training set */
    Mat hist;
    Mat training_set;
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    BOWImgDescriptorExtractor bowextractor(extractor, matcher);
    bowextractor.setVocabulary(vocabulary);

    cout << "Building training set..." << endl;
    for (int i = 0; i < filepaths.size(); i++)
    {
        img = imread(filepaths[i], 0);
        detector->detect(img, keypoints);
        bowextractor.compute(img, keypoints, hist);
        training_set.push_back(hist);
    }

    return 0;
}