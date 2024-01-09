#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
#include <time.h>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <functional>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#define BOOST_PYTHON_STATIC_LIB
#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>

class Node
{
public:
    int id;
    std::vector<Node *> neighbor;
    std::unordered_map<Node *, std::vector<double>> message;
    std::vector<double> prob;

    // エネルギー関数用パラメータ
    double alpha;
    double beta;

    Node(int id) : id(id), alpha(10.0), beta(5.0) {}

    void addNeighbor(Node *node)
    {
        neighbor.push_back(node);
    }

    std::vector<Node *> getNeighbor()
    {
        return neighbor;
    }

    // 隣接ノードからのメッセージを初期化
    void initializeMessage()
    {
        for (Node *neighborNode : neighbor)
        {
            message[neighborNode] = {1.0, 1.0};
        }
    }

    // 全てのメッセージを統合
    // probは周辺分布
    void marginal()
    {
        prob = {1.0, 1.0};

        for (const auto &messageEntry : message)
        {
            const auto &message = messageEntry.second;
            prob[0] *= message[0];
            prob[1] *= message[1];
        }

        double sumProb = prob[0] + prob[1];
        prob[0] /= sumProb;
        prob[1] /= sumProb;
    }

    // 隣接ノードの状態を考慮した尤度を計算
    std::vector<double> sendMessage(Node *target)
    {
        std::vector<double> neighborMessage = {1.0, 1.0};

        for (const auto &neighborNode : neighbor)
        {
            if (neighborNode != target)
            {
                const auto &neighborNodeMessage = message[neighborNode];
                neighborMessage[0] *= neighborNodeMessage[0];
                neighborMessage[1] *= neighborNodeMessage[1];
            }
        }

        std::vector<double> compatibility_0 = {std::exp(-beta * std::abs(0.0 - 0.0)), std::exp(-beta * std::abs(0.0 - 1.0))};
        std::vector<double> compatibility_1 = {std::exp(-beta * std::abs(1.0 - 0.0)), std::exp(-beta * std::abs(1.0 - 1.0))};

        std::vector<double> messageResult = {
            (neighborMessage[0] * compatibility_0[0]) + (neighborMessage[1] * compatibility_1[0]),
            (neighborMessage[0] * compatibility_0[1]) + (neighborMessage[1] * compatibility_1[1])};

        double sumMessageResult = messageResult[0] + messageResult[1];
        messageResult[0] /= sumMessageResult;
        messageResult[1] /= sumMessageResult;

        return messageResult;
    }

    // 観測値から計算する尤度
    void calcLikelihood(int value)
    {
        prob = {0.0, 0.0};

        if (value == 0)
        {
            prob[0] = std::exp(-alpha * 0.0);
            prob[1] = std::exp(-alpha * 1.0);
        }
        else
        {
            prob[0] = std::exp(-alpha * 1.0);
            prob[1] = std::exp(-alpha * 0.0);
        }

        message[this] = prob;
    }
};

class MRF
{
public:
    std::vector<Node *> nodes;          // MRF上のノード
    std::unordered_map<int, Node *> id; // ノードのID

    // MRFにノードを追加する
    void addNode(int i, Node *node)
    {
        nodes.push_back(node);
        this->id[i] = node;
    }

    // IDに応じたノードを返す
    Node *getNode(int i)
    {
        return id[i];
    }

    // 全部のノードを返す
    std::vector<Node *> getNodes()
    {
        return nodes;
    }

    // 確率伝播を開始する
    void beliefPropagation(int iter = 20)
    {
        // 各ノードについて隣接ノードからのメッセージを初期化
        for (Node *node : nodes)
        {
            node->initializeMessage();
        }

        // 一定回数繰り返す
        for (int t = 0; t < iter; ++t)
        {
            std::cout << t << std::endl;

            // 各ノードについて，そのノードに隣接するノードへメッセージを送信する
            for (Node *node : nodes)
            {
                for (Node *neighbor : node->getNeighbor())
                {
                    neighbor->message[node] = node->sendMessage(neighbor);
                }
            }
        }

        // 各ノードについて周辺分布を計算する
        for (Node *node : nodes)
        {
            node->marginal();
        }
    }
};

MRF generateBeliefNetwork(const cv::Mat &image)
{
    MRF network;
    int height = image.rows;
    int width = image.cols;

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int nodeID = width * i + j;
            Node *node = new Node(nodeID); // Assuming Node is a class that you have defined
            network.addNode(nodeID, node);
        }
    }

    int dy[] = {-1, 0, 0, 1};
    int dx[] = {0, -1, 1, 0};

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            Node *node = network.getNode(width * i + j);

            for (int k = 0; k < 4; ++k)
            {
                if (i + dy[k] >= 0 && i + dy[k] < height && j + dx[k] >= 0 && j + dx[k] < width)
                {
                    Node *neighbor = network.getNode(width * (i + dy[k]) + j + dx[k]);
                    node->addNeighbor(neighbor);
                }
            }
        }
    }

    return network;
}

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};
// LBP特徴の画像に変換する関数
void cvt_LBP(const cv::Mat &src, cv::Mat &lbp)
{
    cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    dst = cv::Scalar::all(0);
    cv::Mat padsrc, blur;
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    // cv::bilateralFilter(padsrc, padsrc, 2, 2*2, 2/2);
    // cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);
    // cv::medianBlur(padsrc, blur, 3);
    // padsrc = blur.clone();
    //  cv::imshow("first", src);
    //  cv::imshow("third", lbp);
    for (int x = 1; x < padsrc.cols - 1; x++)
    {
        for (int y = 1; y < padsrc.rows - 1; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= padsrc.at<unsigned char>(y, x))
                        dst.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
                }
            }
        }
    }
    // cv::imshow("second", lbp);
    lbp = dst.clone();
}

void wb_diviser(cv::Mat src, cv::Mat &dst)
{
    dst = cv::Mat(src.rows,src.cols, CV_8UC1);
    cv::Mat copy = src.clone();
    for (int y = 0; y < copy.rows; y++)
    {
        for (int x = 0; x < copy.cols; x++)
        {
            if ((int)copy.at<unsigned char>(y, x) >= 150)
            {
                copy.at<unsigned char>(y, x) = 255;
            }
            else
            {
                copy.at<unsigned char>(y, x) = 0;
            }
        }
    }
    dst = copy.clone();
}
int main()
{
    // 使用データ
    cv::Mat image = cv::imread("images/test_img/left.JPG", cv::IMREAD_GRAYSCALE);
    cv::Mat noise, lbp;
    cv::resize(image, image, cv::Size(1280, 720));
    //noise = image.clone();
    cvt_LBP(image, lbp);
    //image = lbp.clone();
    wb_diviser(lbp, noise);
    //cv::imshow("a", image);
    image = noise.clone();

     //cv::Canny(noise, noise, 10, 100);
    //  MRF構築
    MRF network = generateBeliefNetwork(image); // Assuming generateBeliefNetwork is a function that you have defined

    // 観測値（画素値）から尤度を作成
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Node *node = network.getNode(image.cols * i + j);
            node->calcLikelihood(noise.at<unsigned char>(i, j));
        }
    }

    // 確率伝播法を行う
    network.beliefPropagation();

    // 周辺分布は[0の確率, 1の確率]の順番
    // もし1の確率が大きければoutputの画素値を1に変える
    cv::Mat output = cv::Mat(noise.rows, noise.cols, CV_8UC1);
    output = cv::Scalar::all(0);

    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            Node *node = network.getNode(output.cols * i + j);
            std::vector<double> prob = node->prob;
            if (prob[1] > prob[0])
            {
                output.at<unsigned char>(i, j) = 255;
            }
            else
            {
                output.at<unsigned char>(i, j) = 0;
            }
        }
    }

    // 結果表示
    cv::imshow("Original Noise", noise);
    cv::imshow("Output", output);
    cv::waitKey(0);

    return 0;
}