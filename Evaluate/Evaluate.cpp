#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: Evaluate pos.yml neg.yml" << endl;
        return -1;
    }

    /* Read training samples */
    cout << "Reading training samples..." << endl;
    string samples_file;
    FileStorage fs;
    Mat samples;
    Mat inputs;
    Mat labels;

    samples_file = argv[1]; // read positive samples
    fs.open(samples_file, FileStorage::READ);
    fs[samples_file.substr(0, samples_file.rfind('.'))] >> samples;
    inputs.push_back(samples);
    labels.push_back(Mat(Mat::ones(samples.rows, 1, CV_32F)));

    samples_file = argv[2]; // read negative samples
    fs.open(samples_file, FileStorage::READ);
    fs[samples_file.substr(0, samples_file.rfind('.'))] >> samples;
    inputs.push_back(samples);
    labels.push_back(Mat(Mat::zeros(samples.rows, 1, CV_32F)));

    samples.release();
    fs.release();

    /* Neural Network */
    cout << "Reading trained neural network..." << endl;
    CvANN_MLP nn;
    nn.load("nn.yml");

    Mat resp;
    nn.predict(inputs, resp);

    Mat preds;
    threshold(resp, preds, 0, 1, THRESH_BINARY);

    Mat diff;
    compare(labels, preds, diff, CMP_NE);
    int nz = countNonZero(diff);

    return 0;
}