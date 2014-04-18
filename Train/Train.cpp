#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: Train pos_samples.yml neg_samples.yml" << endl;
        return -1;
    }

    /* Read training samples */
    cout << "Reading training samples..." << endl;
    string samples_file;
    FileStorage fs;
    Mat samples;
    Mat inputs;
    Mat pos_labels;
    Mat neg_labels;
    Mat labels;

    samples_file = argv[1]; // read positive samples
    fs.open(samples_file, FileStorage::READ);
    fs[samples_file.substr(0, samples_file.rfind('.'))] >> samples;
    inputs.push_back(samples);
    hconcat(Mat::ones(samples.rows, 1, CV_32F), Mat(samples.rows, 1, CV_32F, Scalar(-1)), pos_labels);

    samples_file = argv[2]; // read negative samples
    fs.open(samples_file, FileStorage::READ);
    fs[samples_file.substr(0, samples_file.rfind('.'))] >> samples;
    inputs.push_back(samples);
    hconcat(Mat(samples.rows, 1, CV_32F, Scalar(-1)), Mat::ones(samples.rows, 1, CV_32F), neg_labels);

    vconcat(pos_labels, neg_labels, labels);

    samples.release();
    pos_labels.release();
    neg_labels.release();
    fs.release();

    /* Neural network */
    cout << "Constructing neural network..." << endl;
    CvANN_MLP_TrainParams params;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01);
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.1;
    params.bp_moment_scale = 0.1;

    Mat layerSizes = (Mat_<int>(1, 3) << 50, 40, 2);
    CvANN_MLP nn(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1); // activation function: alpha=1, beta=1

    cout << "Training neural network..." << endl;
    nn.train(inputs, labels, Mat(), Mat(), params);

    string model_file = "nn.yml";
    cout << "Save neural network to " << model_file << "..." << endl;
    nn.save("nn.yml");

    cout << "Done." << endl;

    return 0;
}