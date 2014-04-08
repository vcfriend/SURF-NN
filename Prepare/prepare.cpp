#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree.hpp>

using namespace std;
using namespace cv;

void progress(int i, size_t size)
{
        if (i != 0) cout << "\b\b\b";
        cout << setw(2) << i * 100 / size << "%";
        if (i == size) cout << endl;
}

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

    cout << "Extracting descriptors...";
    for (int i = 0; i < filepaths.size(); i++)
    {
        img = imread(filepaths[i], 0);
        detector->detect(img, keypoints);
        extractor->compute(img, keypoints, descriptors);
        all_descriptors.push_back(descriptors);
        progress(i, filepaths.size() - 1);
    }

    cout << "Building vocabulary..." << endl;
    Mat vocabulary = bowtrainer.cluster(all_descriptors);

    /* build training set */
    Mat hist;
    Mat training_set;
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    BOWImgDescriptorExtractor bowextractor(extractor, matcher);
    bowextractor.setVocabulary(vocabulary);

    cout << "Building training set...";
    for (int i = 0; i < filepaths.size(); i++)
    {
        img = imread(filepaths[i], 0);
        detector->detect(img, keypoints);
        bowextractor.compute(img, keypoints, hist);
        training_set.push_back(hist);
        progress(i, filepaths.size() - 1);
    }

    FileStorage fs("training_set.yml", FileStorage::WRITE);
    fs << "training_set" << training_set;
    fs.release();

    return 0;
}