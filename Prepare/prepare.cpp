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

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: Prepare image_list samples_file" << endl;
        return -1;
    }

    string samples_file(argv[2]);

    /* get filepaths vector */
    fstream filelist(argv[1]);
    vector<string> filepaths;
    string filepath;

    while (getline(filelist, filepath))
        filepaths.push_back(filepath);

    /* build BOW vocabulary */
    Mat img;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat all_descriptors;
    Ptr<FeatureDetector> detector(new SurfFeatureDetector(100));
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
    BOWKMeansTrainer bowtrainer(50);

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
    Mat samples;
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    BOWImgDescriptorExtractor bowextractor(extractor, matcher);
    bowextractor.setVocabulary(vocabulary);

    cout << "Building training set...";
    for (int i = 0; i < filepaths.size(); i++)
    {
        img = imread(filepaths[i], 0);
        detector->detect(img, keypoints);
        bowextractor.compute(img, keypoints, hist);
        samples.push_back(hist);
        progress(i, filepaths.size() - 1);
    }

    cout << "Writing to " << samples_file << ".yml..." << endl;
    FileStorage fs(samples_file + ".yml", FileStorage::WRITE);
    fs << samples_file << samples;
    fs.release();
    cout << "Done." << endl;

    return 0;
}