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

    /* get BOW vocabulary */
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
    else
    {
        cout << "Vocabulary file not found." << endl;
        cout << "Building vocabulary..." << endl;
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

        cout << "Clustering descriptors..." << endl;
        vocabulary = bowtrainer.cluster(all_descriptors);

        cout << "Writing to vocabulary.yml..." << endl;
        fs.open("vocabulary.yml", FileStorage::WRITE);
        fs << "vocabulary" << vocabulary;

        fs.release();
        all_descriptors.release();
    }

    /* build training set */
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    BOWImgDescriptorExtractor bowextractor(extractor, matcher);
    bowextractor.setVocabulary(vocabulary);

    cout << "Building training set...";
    for (int i = 0; i < filepaths.size(); i++)
    {
        img = imread(filepaths[i], 0);
        detector->detect(img, keypoints);
        bowextractor.compute(img, keypoints, descriptors);
        all_descriptors.push_back(descriptors);
        progress(i, filepaths.size() - 1);
    }

    cout << "Writing to " << samples_file << ".yml..." << endl;
    fs.open(samples_file + ".yml", FileStorage::WRITE);
    fs << samples_file << all_descriptors;
    fs.release();

    cout << "Done." << endl;

    return 0;
}