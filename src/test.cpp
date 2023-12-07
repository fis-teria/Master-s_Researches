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

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for)
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif
#else
#define OMP_PARALLEL_FOR
#endif

std::string dir = "images/Tsukuba0";
std::string tag = ".jpg";
int DB_dir_num = 0;
int Cam_dir_num = 2;

const int WIDTH = 896;  // 1280 896
const int HEIGHT = 504; // 720 504
int D_MAG = 15;         // H = 距離ｘ倍率(H<150)  ex) 最長距離を10mにしたければ倍率を15にすればよい
static std::mutex m;

int FOCUS = 24;                                              // 焦点 mm
float IS_WIDTH = 4.8;                                        // 撮像素子の横 1/3レンズなら4.8mm
float IS_HEIGHT = 3.6;                                       // 撮像素子の縦 1/3レンズなら3.6mm
float PXL_WIDTH = (IS_WIDTH / WIDTH);                        // 1pixelあたりの横の長さ mm
float PXL_HEIGHT = (IS_HEIGHT / HEIGHT);                     // 1pixelあたりの縦の長さ mm
int CAM_DIS = 10;                                            // カメラ間の距離 10cm
double D_CALI = (FOCUS * CAM_DIS * 10) / (PXL_WIDTH * 1000); // 距離を求めるのに必要な定数項 mで換算 式(焦点(mm) * カメラ間距離(cm))/(1pixel長(mm)*画像内の距離)

const int NTSS_GRAY = 0;
const int NTSS_RGB = 1;

const int L2R = -1;
const int R2L = 1;

const int debug = 1;
const int sim_BM_check = 3;
const int vec_check = 2;
const int thread_check = 1;

const int BLOCK_MODE = NTSS_GRAY;
constexpr size_t ThreadCount = 8;
template <size_t Count>
class worker_pool
{
public:
    worker_pool()
    {
        int index = 0;
        for (auto &inner : inner_workers_)
        {
            inner.initialize(this, index++);
        }
    }
    ~worker_pool()
    {
        wait_until_idle();
        request_termination();
    };

    template <typename F>
    void run(F &&task)
    {
        auto current_thread_index = get_current_thread_index();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // インデックス-1はワーカー外スレッドを指す
            if (current_thread_index == -1)
            {
                global_queue_.emplace_back(std::forward<F>(task));
            }
            else
            {
                inner_workers_[current_thread_index].push(std::forward<F>(task));
            }
        }
        wakeup_all(current_thread_index);
    }

    void wait_until_idle()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]()
                       { return global_queue_.empty() || is_requested_termination; });
        }
        for (auto &inner : inner_workers_)
        {
            inner.wait_until_idle();
        }
    }

    void request_termination()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_requested_termination = true;
            cond_.notify_all();
        }
        for (auto &inner : inner_workers_)
        {
            inner.request_termination();
        }
    }

    std::function<void()> steal_or_pull(int index)
    {
        for (int i = 0; i < Count; ++i)
        {
            if (i == index)
            {
                continue;
            }
            auto task = inner_workers_[i].steal();
            if (!!task)
            {
                return std::move(task);
            }
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (global_queue_.empty())
        {
            return {};
        }

        auto task = global_queue_.front();
        global_queue_.pop_front();
        cond_.notify_all();
        return std::move(task);
    }

    void wakeup_all(int index)
    {
        int i = 0;
        for (auto &inner : inner_workers_)
        {
            if (i != index)
            {
                inner.wakeup();
            }
            ++i;
        }
    }

private:
    class inner_worker
    {
    public:
        inner_worker() : thread_([this]()
                                 { proc_worker(); }) {}
        ~inner_worker()
        {
            wait_until_idle();
            request_termination();
            if (thread_.joinable())
            {
                thread_.join();
            }
        }

        void initialize(worker_pool *parent, int index)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            parent_ = parent;
            index_ = index;
            cond_.notify_all();
        }

        template <typename F>
        void push(F &&task)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            local_queue_.emplace_back(task);
        }

        std::function<void()> steal()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (local_queue_.empty())
            {
                return {};
            }

            auto task = local_queue_.front();
            local_queue_.pop_front();
            cond_.notify_all();
            return std::move(task);
        }

        void wakeup()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.notify_all();
        }

        void wait_until_idle()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]()
                       { return (local_queue_.empty() && !current_task_) || is_requested_termination; });
        }

        void request_termination()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_requested_termination = true;
            cond_.notify_all();
        }

        std::thread::id get_thread_id() const
        {
            return thread_.get_id();
        }

    private:
        void wait_initialize()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]()
                       { return parent_ != nullptr && index_ >= 0; });
        }

        void proc_worker()
        {
            wait_initialize();

            while (true)
            {
                bool is_assigned = false;
                {
                    // ローカルキュー末尾からの取り出しを優先する
                    std::unique_lock<std::mutex> lock(mutex_);
                    if (!local_queue_.empty())
                    {
                        current_task_ = local_queue_.back();
                        local_queue_.pop_back();
                        is_assigned = true;
                    }
                }
                if (!is_assigned)
                {
                    auto task = parent_->steal_or_pull(index_);
                    if (!task)
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        current_task_ = {};
                        cond_.notify_all(); // 何も処理していないことを通知する
                        cond_.wait(lock, [&]()
                                   {
                            // この述語内ではロックを取得しているはず
                            if (is_requested_termination) {
                                return true;
                            }
                            // steal_or_pullの時にロックを保持しているとデッドロックを起こす
                            lock.unlock();
                            auto task = parent_->steal_or_pull(index_);
                            lock.lock();
                            // current_task_の更新時に再度ロックを取得する
                            current_task_ = std::move(task);
                            return !!current_task_; });
                        if (is_requested_termination)
                        {
                            break;
                        }
                    }
                    else
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        current_task_ = std::move(task);
                    }
                }
                current_task_();
            }
        }

        worker_pool *parent_{nullptr};
        int index_{-1};
        std::function<void()> current_task_{};
        bool is_requested_termination{false};
        std::thread thread_;
        std::deque<std::function<void()>> local_queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
    };

    int get_current_thread_index()
    {
        auto current_id = std::this_thread::get_id();
        int index = 0;
        for (auto &worker : inner_workers_)
        {
            if (current_id == worker.get_thread_id())
            {
                return index;
            }
            ++index;
        }
        return -1;
    }

    inner_worker inner_workers_[Count];
    bool is_requested_termination{false};
    std::deque<std::function<void()>> global_queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

// 事前走行画像が入ったフォルダから画像のパスをとってくる関数
std::string make_tpath(std::string dir, int dir_num, int var, std::string tag)
{
    std::string back;
    if (var < 10)
    {
        back = dir + std::to_string(dir_num) + "/00000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 100)
    {
        back = dir + std::to_string(dir_num) + "/0000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 1000)
    {
        back = dir + std::to_string(dir_num) + "/000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 10000)
    {
        back = dir + std::to_string(dir_num) + "/00" + std::to_string(var) + tag;
        return back;
    }
}

std::string make_spath(std::string dir, int var, std::string tag)
{
    std::string back;
    if (var < 10)
    {
        back = dir + "/00000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 100)
    {
        back = dir + "/0000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 1000)
    {
        back = dir + "/000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 10000)
    {
        back = dir + "/00" + std::to_string(var) + tag;
        return back;
    }
}

void print_elapsed_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %15.15f sec\n", elapsed);
}

// LBP特徴にかけるフィルター
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

void detective()
{
    cv::Mat lbp, edge, det;
    std::string numf = dir + std::to_string(Cam_dir_num) + "/num.txt";
    std::string line;
    int num = 0;
    std::ifstream ifs(numf);
    while (getline(ifs, line))
    {
        num = std::stoi(line);
    }
    std::cout << "start" << std::endl;
    std::cout << num << std::endl;
    for (int i = 339; i < 340; i++)
    {
        clock_t begin = clock();
        std::string path = make_tpath(dir, Cam_dir_num, i, tag);
        std::cout << path << std::endl;
        cv::Mat img = cv::imread(path, 0);
        if (img.empty())
        {
            std::cout << "not found images" << std::endl;
            break;
        }

        cvt_LBP(img, lbp);

        cv::Canny(img, edge, 100, 200);

        // cv::findContours は第一引数を破壊的に利用するため imshow 用に別変数を用意しておきます。
        cv::Mat canny2 = edge.clone();

        // cv::Point の配列として、輪郭を計算します。
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(canny2, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        std::cout << contours.size() << std::endl;                  //=> 36
        std::cout << contours[contours.size() - 1][0] << std::endl; //=> [154, 10]

        // 輪郭を可視化してみます。分かりやすさのため、乱数を利用して色付けします。
        cv::Mat drawing = cv::Mat::zeros(canny2.size(), CV_8UC3);
        cv::RNG rng(12345);

        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            cv::drawContours(drawing, contours, (int)i, color);
        }

        det = drawing.clone();
        // cv::add(edge, lbp, det);
        clock_t end = clock();
        print_elapsed_time(begin, end);

        cv::imshow("hog", img);
        cv::imshow("lbp", lbp);
        cv::imshow("canny", edge);
        cv::imshow("result", det);
        // cv::imwrite("output.jpg", img);
        cv::waitKey(0);
    }
}

// キャリブレーションから得られた外部関数を読み込むクラス
class readXml
{
public:
    cv::Mat camera_matrix, distcoeffs;
    std::string FILE_NAME;

public:
    readXml()
    {
    }

public:
    readXml(std::string f_name)
    {
        cv::FileStorage fs(f_name, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
        std::cout << "constractor" << std::endl;
    }

public:
    readXml(const readXml &xml)
    {
        std::cout << "copy constractor" << std::endl;
    }

public:
    void read(std::string f_name)
    {
        cv::FileStorage fs(f_name, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
        std::cout << "member function" << std::endl;
    }
};

// ブロックマッチングに使うかもしれない構造体
class BM
{
public:
    int x;
    int y;
    int sum;

public:
    BM()
    {
    }

public:
    BM(const int origin_x, const int origin_y, const int s)
    {
        x = origin_x;
        y = origin_y;
        sum = s;
    }

public:
    BM(const BM &BM)
    {
        x = BM.x;
        y = BM.y;
        sum = BM.sum;
    }

public:
    void get_ELEMENTS()
    {
        std::cout << "x, y, sum "
                  << " " << x << " " << y << " " << sum << std::endl;
    }
};

// シンプルなブロックマッチング グレースケール
double sim_G_BM(const cv::Mat &block, const cv::Mat &src, int origin_x, int origin_y, int LorR)
{
    if (debug == sim_BM_check)
        std::cout << "start sim_BM" << std::endl;

    cv::Mat rect = src.clone();
    std::vector<BM> match_Result;
    int dist;
    double depth;
    int BM_size = 0;
    int sum = 0;
    int y = origin_y;

    int start_x = 0;
    int search_lange = 0;
    int end_lange = 0;

    if (LorR == R2L)
    {
        start_x = origin_x - 100;
        search_lange = WIDTH / 2;
    }
    else if (LorR == L2R)
    {
        start_x = origin_x - WIDTH / 2;
        search_lange = 100;
    }
    end_lange = origin_x + search_lange;

    if (start_x < 0)
        start_x = 0;
    if (end_lange > src.cols)
        end_lange = src.cols;

    for (int x = start_x; x < end_lange; x += block.cols)
    {
        if (x < src.cols)
        {
            // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
            // match_Result.resize(BM_size + 1);
            // match_Result[BM_size].x = x;
            // match_Result[BM_size].y = y;

            // std::cout << "start block matching" << std::endl;
            for (int i = 0; i < block.cols; i++)
            {
                for (int j = 0; j < block.rows; j++)
                {
                    if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                    {
                        // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                        sum += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                    }
                }
            }
            // match_Result[BM_size].sum = sum;
            match_Result.push_back(BM(x, y, sum));
            // std::cout << "sum " << sum << std::endl;
            sum = 0;
            BM_size++;
        }
    }

    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });

    if (debug == sim_BM_check)
        std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size();

    /*
    int sx = match_Result[0].x;
    int sy = match_Result[0].y;
    BM_size = 0;
    match_Result.resize(BM_size);
    // std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size() << std::endl;

    for (int x = sx - block.cols; x < sx + block.cols; x++)
    {
        if (x < src.cols && x > -1)
        {
            // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
            match_Result.resize(BM_size + 1);
            match_Result[BM_size].x = x;
            match_Result[BM_size].y = y;

            // std::cout << "start block matching" << std::endl;
            for (int i = 0; i < block.cols; i++)
            {
                for (int j = 0; j < block.rows; j++)
                {
                    if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                    {
                        // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                        sum += abs(src.at<cv::Vec3b>(y + j, x + i)[0] - block.at<cv::Vec3b>(j, i)[0]) + abs(src.at<cv::Vec3b>(y + j, x + i)[1] - block.at<cv::Vec3b>(j, i)[1]) + abs(src.at<cv::Vec3b>(y + j, x + i)[2] - block.at<cv::Vec3b>(j, i)[2]);
                    }
                }
            }
            match_Result[BM_size].sum = sum;
            // std::cout << "sum " << sum << std::endl;
            sum = 0;
            BM_size++;
        }
    }

    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });
    //*/
    dist = sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) - (origin_y - match_Result[0].y) * (origin_y - match_Result[0].x)));
    if (debug == sim_BM_check)
        std::cout << " distance = " << dist;

    depth = D_CALI / dist;
    if (debug == sim_BM_check)
        std::cout << " depth = " << depth << std::endl;

    return depth;
}
// シンプルなブロックマッチング BGR対応
double sim_C_BM(const cv::Mat &block, const cv::Mat &src, int origin_x, int origin_y, int LorR)
{
    if (debug == sim_BM_check)
        std::cout << "start sim_BM" << std::endl;

    cv::Mat rect = src.clone();
    std::vector<BM> match_Result;
    int dist;
    double depth;
    int BM_size = 0;
    int sum = 0;
    int y = origin_y;

    int start_x = 0;
    int search_lange = 0;
    int end_lange = 0;

    if (LorR == R2L)
    {
        start_x = origin_x - 100;
        search_lange = WIDTH / 2;
    }
    else if (LorR == L2R)
    {
        start_x = origin_x - WIDTH / 2;
        search_lange = 100;
    }
    end_lange = origin_x + search_lange;

    if (start_x < 0)
        start_x = 0;
    if (end_lange > src.cols)
        end_lange = src.cols;

    for (int x = start_x; x < end_lange; x += block.cols)
    {
        if (x < src.cols)
        {
            // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
            // match_Result.resize(BM_size + 1);
            // match_Result[BM_size].x = x;
            // match_Result[BM_size].y = y;
            /*
            rect = src.clone();
            cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
            cv::imshow("dd", rect);
            const int key = cv::waitKey(10);
            //*/

            // std::cout << "start block matching" << std::endl;
            for (int i = 0; i < block.cols; i++)
            {
                for (int j = 0; j < block.rows; j++)
                {
                    if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                    {
                        // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                        sum += abs(src.at<cv::Vec3b>(y + j, x + i)[0] - block.at<cv::Vec3b>(j, i)[0]) + abs(src.at<cv::Vec3b>(y + j, x + i)[1] - block.at<cv::Vec3b>(j, i)[1]) + abs(src.at<cv::Vec3b>(y + j, x + i)[2] - block.at<cv::Vec3b>(j, i)[2]);
                    }
                }
            }
            // match_Result[BM_size].sum = sum;
            match_Result.push_back(BM(x, y, sum));

            // std::cout << "sum " << sum << std::endl;
            sum = 0;
            BM_size++;
        }
    }

    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });

    ///*
    int sx = match_Result[0].x;
    int sy = match_Result[0].y;
    BM_size = 0;
    match_Result.resize(BM_size);
    // std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size() << std::endl;

    for (int x = sx - block.cols; x < sx + block.cols; x++)
    {
        if (x < src.cols && x > -1)
        {
            // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
            match_Result.resize(BM_size + 1);
            match_Result[BM_size].x = x;
            match_Result[BM_size].y = y;

            // std::cout << "start block matching" << std::endl;
            for (int i = 0; i < block.cols; i++)
            {
                for (int j = 0; j < block.rows; j++)
                {
                    if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                    {
                        // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                        sum += abs(src.at<cv::Vec3b>(y + j, x + i)[0] - block.at<cv::Vec3b>(j, i)[0]) + abs(src.at<cv::Vec3b>(y + j, x + i)[1] - block.at<cv::Vec3b>(j, i)[1]) + abs(src.at<cv::Vec3b>(y + j, x + i)[2] - block.at<cv::Vec3b>(j, i)[2]);
                    }
                }
            }
            match_Result[BM_size].sum = sum;
            // std::cout << "sum " << sum << std::endl;
            sum = 0;
            BM_size++;
        }
    }

    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });
    //*/
    dist = sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) - (origin_y - match_Result[0].y) * (origin_y - match_Result[0].x)));
    // std::cout << " distance = " << dist << std::endl;

    depth = D_CALI / dist;
    // std::cout << "depth = " << depth << std::endl;

    return depth;
}

// グレースケール画像のブロックマッチング アルゴリズムはNTSS法を使用
/*
double NTSS(const cv::Mat &block, const cv::Mat &src, int origin_x, int origin_y, int step)
{
    cv::Mat rect = src.clone();
    std::vector<BM> match_Result;
    int dist;
    double depth;
    int BM_size = 0;
    int sum = 0;
    int k = origin_x - step;
    // if (k < 0)
    //     k = 0;
    int l = origin_y - step;
    // if (l < 0)
    //     l = 0;

    int end_x = origin_x + step;
    if (end_x > src.cols)
        end_x = src.cols;
    int end_y = origin_y + step;
    if (end_y > src.rows)
        end_y = src.rows;
    // std::cout << "before error" << std::endl;
    // std::cout << x << " " << y << std::endl;

    // first step
    // step distance round search
    // std::cout << "origin_x " << origin_x << " origin_y " << origin_y << std::endl;
    // std::cout << "first step" << std::endl;
    // std::cout << "step distance round search" << std::endl;

    for (int x = k; x <= end_x; x += step)
    {
        for (int y = l; y <= end_y; y += step)
        {
            if (x == origin_x && y == origin_y)
            {
                continue;
            }
            else
            {
                // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = x;
                match_Result[BM_size].y = y;

rect = src.clone();
                cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
                cv::imshow("dd", rect);
                const int key = cv::waitKey(10);

                // std::cout << "start block matching" << std::endl;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                        {
                            // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                            sum += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sum = sum;
                // std::cout << "sum " << sum << std::endl;
                sum = 0;
                BM_size++;
            }
        }
    }

    // origin round search
    // std::cout << "origin round search" << std::endl;
    for (int x = origin_x - 1; x <= origin_x + 1; x++)
    {
        for (int y = origin_y - 1; y <= origin_y + 1; y++)
        {
            if (x >= 0 && y >= 0 && x <= src.cols && y <= src.rows)
            {
                // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = x;
                match_Result[BM_size].y = y;

rect = src.clone();
                cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
                cv::imshow("dd", rect);
                const int key = cv::waitKey(10);

                // std::cout << "start block matching" << std::endl;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                        {
                            // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                            sum += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sum = sum;
                // std::cout << "sum " << sum << std::endl;
                sum = 0;
                BM_size++;
            }
        }
    }
    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });
    // std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size() << std::endl;

    int sx = match_Result[0].x;
    int sy = match_Result[0].y;
    int tx = origin_x;
    int ty = origin_y;

    // std::cout << "origin point (" << origin_x << " " << origin_y << ") -> first match(" << sx << " " << sy << ")";

    //  second steps and more
    for (int sstep = step / 2; sstep > 0; sstep /= 2)
    {
        match_Result.resize(0);
        BM_size = 0;
        if ((sx - tx) * (sx - tx) + (sy - ty) * (sy - ty) > 2)
        {
            // std::cout << "far distance" << std::endl;
            for (int x = sx - sstep; x <= sx + sstep; x += sstep)
            {
                for (int y = sy - sstep; y <= sy + sstep; y += sstep)
                {
                    if (x == origin_x && y == origin_y)
                    {
                        continue;
                    }
                    else
                    {
                        // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                        match_Result.resize(BM_size + 1);
                        match_Result[BM_size].x = x;
                        match_Result[BM_size].y = y;

rect = src.clone();
                        cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
                        cv::imshow("dd", rect);
                        const int key = cv::waitKey(10);

                        // std::cout << "start block matching" << std::endl;
                        for (int i = 0; i < block.cols; i++)
                        {
                            for (int j = 0; j < block.rows; j++)
                            {
                                if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                                {
                                    // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                                    sum += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                                }
                            }
                        }
                        match_Result[BM_size].sum = sum;
                        // std::cout << "sum " << sum << std::endl;
                        sum = 0;
                        BM_size++;
                    }
                }
            }
            std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
                      { return alpha.sum < beta.sum; });
            // std::cout << "origin point (" << sx << " " << sy << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size() << std::endl;
            // std::cout << " -> match(" << match_Result[0].x << " " << match_Result[0].y << ")";
        }
        else
        {
            // std::cout << "near distance" << std::endl;

            //    A P O N M
            //    B 1 4 7 L
            //    C 2 5 8 K
            //    D 3 6 9 J
            //    E F G H I

            if (abs(sx - origin_x) < 0)
            {
                if (abs(sy - origin_y) < 0)
                {
                    // std::cout << "match left up" << std::endl;
                    //  1
                    //  A
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // P
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // B
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
                else if (abs(sy - origin_y) == 0)
                {
                    // std::cout << "match left" << std::endl;
                    //  2
                    //  B
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // C
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // D
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
                else if (abs(sy - origin_y) > 0)
                {
                    // std::cout << "match left down" << std::endl;
                    //  3
                    //  D
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // E
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // F
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
            }
            else if (abs(sx - origin_x) == 0)
            {
                if (abs(sy - origin_y) < 0)
                {
                    // std::cout << "match up" << std::endl;
                    //  4
                    //  P
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // O
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // N
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
                else if (abs(sy - origin_y) == 0)
                {
                    // std::cout << "match second origin" << std::endl;
                }
                else if (abs(sy - origin_y) > 0)
                {
                    // std::cout << "match down" << std::endl;
                    //  6
                    //  F
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx - 1;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // G
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // H
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
            }
            else if (abs(sx - origin_x) > 0)
            {
                if (abs(sy - origin_y) < 0)
                {
                    // std::cout << "match right up" << std::endl;
                    //  7
                    //  N
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // M
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // L
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
                else if (abs(sy - origin_y) == 0)
                {
                    // std::cout << " match right" << std::endl;
                    //  8
                    //  L
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy - 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // K
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // J
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
                else if (abs(sy - origin_y) > 0)
                {
                    // std::cout << "match right down" << std::endl;
                    //  9
                    //  H
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // I
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy + 1;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                    // J
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = sx + 1;
                    match_Result[BM_size].y = sy;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                            {
                                // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                                sum += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sum = sum;
                    // std::cout << sum << std::endl;
                    sum = 0;
                    BM_size++;
                }
            }
            std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
                      { return alpha.sum < beta.sum; });
            // std::cout << "origin point (" << sx << " " << sy << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size() << std::endl;
            // std::cout << " -> second match(" << match_Result[0].x << " " << match_Result[0].y << ")     distance = " << sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) - (origin_y - match_Result[0].y) * (origin_y - match_Result[0].y))) << std::endl;
            dist = sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) - (origin_y - match_Result[0].y) * (origin_y - match_Result[0].x)));
            // std::cout << " distance = " << dist << std::endl;

            depth = (16 * 1000 * 10 * 10 * 1000) / (6.25 * dist) / 1000000;
            // std::cout << "depth = " << depth << std::endl;

            return depth;
        }
        tx = sx;
        ty = sy;
        sx = match_Result[0].x;
        sy = match_Result[0].y;
    }
    dist = sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) - (origin_y - match_Result[0].y) * (origin_y - match_Result[0].x)));
    // std::cout << " distance = " << dist << std::endl;

    depth = (24 * 1000 * 7 * 10 * 1000) / (6.25 * dist) / 1000000;
    // std::cout << "depth = " << depth << std::endl;

    return depth;
}
*/

// ブロックマッチングの結果を保持するマルチスレッド用のクラス
class BLOCK_MATCHING
{
public:
    int origin_x, origin_y;
    double depth;

public:
    BLOCK_MATCHING(const int x, const int y, const double d)
    {
        origin_x = x;
        origin_y = y;
        depth = d;
    }

    BLOCK_MATCHING(const BLOCK_MATCHING &B_M)
    {
        origin_x = B_M.origin_x;
        origin_y = B_M.origin_y;
        depth = B_M.depth;
    }

public:
    void get_ELEMENTS()
    {
        std::cout << "x, y, depth "
                  << " " << origin_x << " " << origin_y << " " << depth << std::endl;
    }
};

// 深度マップから元画像ないの特定の距離のものを抽出する。
void get_depth(const cv::Mat &src, cv::Mat &dst, const cv::Mat &origin)
{
    cv::Mat copy = cv::Mat(src.rows, src.cols, CV_8UC1);
    copy = cv::Scalar::all(0);
    int H;

    for (int y = 0; y < copy.rows; y++)
    {
        for (int x = 0; x < copy.cols; x++)
        {
            H = src.at<cv::Vec3b>(y, x)[0];
            if (H / D_MAG > 1)
                copy.at<unsigned char>(y, x) = 0;
            else
            {
                copy.at<unsigned char>(y, x) = origin.at<unsigned char>(y, x);
            }
        }
    }
    dst = copy.clone();
}

// スレッドプール
worker_pool<ThreadCount> worker;

// ブロックマッチングの準備をする関数
void block_Matching(cv::Mat &block, const cv::Mat &src, int block_size, int mode, int LorR)
{
    int times = 0;
    int sum = 4;

    std::vector<int> time;
    std::vector<BLOCK_MATCHING> vec_bm;
    cv::Mat depth_map = cv::Mat(block.rows, block.cols, CV_8UC3);
    depth_map = cv::Scalar::all(0);
    cv::Mat depth_map_HSV = depth_map.clone();
    int b_size = block_size;
    if (b_size % 2 != 1)
        b_size++;

    int depth_H = 0;
    int x_count = 0;
    int y_count = 0;

    clock_t begin = clock();
    // マルチスレッド
    /*
    for (int end_cols = block.cols / 2; end_cols <= block.cols; end_cols += block.cols / 2)
    {
        y_count = 0;
        for (int end_rows = block.rows / 2; end_rows <= block.rows; end_rows += block.rows / 2)
        {
            clock_t s = clock();
            worker.run([mode, b_size, &vec_bm, end_cols, end_rows, &times, block, src, s, x_count, y_count, LorR]()
                       {
                            if (debug == thread_check)
                            {
                                std::cout << " thread start" << std::endl;
                            }

                           clock_t begin = clock();
                           // std::cout << "start time " << (float)(begin - s)/CLOCKS_PER_SEC << " sec" << std::endl;
                           // std::cout << "end_cols, end_rows, x_count, y_count " << end_cols << " " << end_rows << " " << x_count << " " << y_count<< std::endl;
                           double depth = 0;
                           if (mode == 0)
                               for (int y = b_size / 2 + (block.rows / 2 - 2) * y_count; y <= end_rows; y += b_size)
                               {
                                   for (int x = b_size / 2 + (block.cols / 2 - 2) * x_count; x <= end_cols; x += b_size)
                                   {
                                       if (y + b_size / 2 < end_rows && x + b_size / 2 < end_cols)
                                       {
                                           // times++;
                                           depth = sim_G_BM(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, LorR);
                                           // depth = NTSS(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, 32);
                                           m.lock();
                                           vec_bm.push_back(BLOCK_MATCHING(x, y, depth));
                                           m.unlock();
                                       }
                                   }
                               }
                           else if (mode == 1)
                               for (int y = b_size / 2 + (block.rows / 2 - 3) * y_count; y <= end_rows; y += b_size)
                               {
                                   for (int x = b_size / 2 + (block.cols / 2 - 2) * x_count; x <= end_cols; x += b_size)
                                   {
                                       if (y + b_size / 2 < end_rows && x + b_size / 2 < end_cols)
                                       {
                                           // times++;
                                           depth = sim_C_BM(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, LorR);
                                           // depth = NTSS(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, 32);
                                           m.lock();
                                           vec_bm.push_back(BLOCK_MATCHING(x, y, depth));
                                           m.unlock();
                                       }
                                   }
                               }
                           m.lock();
                           times++;
                           if(debug == thread_check)
                           {
                            std::cout << "times = " << times << std::endl;
                           }
                           m.unlock();
                           clock_t end = clock();
                           float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
                           printf("Elapsed Time: %15.7f sec\n", elapsed); });
            clock_t e = clock();
            print_elapsed_time(s, e);
            y_count++;
        }
        x_count++;
    }
    if (debug == vec_check)
        std::cout << vec_bm.size() << std::endl;

    while (1){
        //if(sum == times)break;
        //std::cout << sum << " " << times << std::endl;
        //std::cout << "wait time" << std::endl;
    }

    for (int i = 0; i < vec_bm.size(); i++)
    {
        depth_H = vec_bm[i].depth * 20;
        if (depth_H > 150)
            depth_H = 150;
        if (depth_H < 0)
            depth_H = 0;

        if (debug == vec_check)
            vec_bm[i].get_ELEMENTS();
        // std::cout << vec_bm[i].depth << std::endl;
        cv::rectangle(depth_map, cv::Point(vec_bm[i].origin_x - b_size / 2, vec_bm[i].origin_y - b_size / 2), cv::Point(vec_bm[i].origin_x + b_size / 2, vec_bm[i].origin_y + b_size / 2), cv::Scalar(depth_H, 255, 255), cv::FILLED);
    }
    std::cout << "synchronous" << std::endl;
    //*/

    // シングルスレッド
    ///*
    std::cout << "single thread" << std::endl;
    double depth = 0;
    if (mode == 0)
        OMP_PARALLEL_FOR
#pragma omp private(depth_H, depth, x, block, src)
    for (int x = b_size / 2; x < block.cols; x += b_size)
    {
        for (int y = b_size / 2; y < block.rows; y += b_size)
        {
            if (y + b_size / 2 < block.rows && x + b_size / 2 < block.cols)
            {
                times++;
                depth = sim_G_BM(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, LorR);
#pragma omp critical
                vec_bm.push_back(BLOCK_MATCHING(x, y, depth));
            }
        }
    }
    else if (mode == 1)
        OMP_PARALLEL_FOR
#pragma omp private(depth_H, depth, x, block, src)
        for (int x = b_size / 2; x < block.cols; x += b_size)
    {
        for (int y = b_size / 2; y < block.rows; y += b_size)
        {
            if (y + b_size / 2 < block.rows && x + b_size / 2 < block.cols)
            {
                times++;
                depth = sim_C_BM(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, LorR);
#pragma omp critical
                vec_bm.push_back(BLOCK_MATCHING(x, y, depth));
            }
        }
    }
    std::cout << "end loop" << std::endl;
    //*/

    clock_t end = clock();
    print_elapsed_time(begin, end);

    for (int i = 0; i < vec_bm.size(); i++)
    {
        depth_H = vec_bm[i].depth * D_MAG;
        if (depth_H > 150)
            depth_H = 150;
        if (depth_H < 0)
            depth_H = 0;

        if (debug == vec_check)
            vec_bm[i].get_ELEMENTS();
        // std::cout << vec_bm[i].depth << std::endl;
        cv::rectangle(depth_map, cv::Point(vec_bm[i].origin_x - b_size / 2, vec_bm[i].origin_y - b_size / 2), cv::Point(vec_bm[i].origin_x + b_size / 2, vec_bm[i].origin_y + b_size / 2), cv::Scalar(depth_H, 255, 255), cv::FILLED);
    }
    std::cout << "synchronous" << std::endl;

    cv::cvtColor(depth_map, depth_map_HSV, cv::COLOR_HSV2BGR);
    cv::Mat nearline;
    get_depth(depth_map_HSV, nearline, block);

    if (LorR == R2L)
    {
        cv::imshow("depth R2L", depth_map_HSV);
        cv::imshow("depth R2L nearline", nearline);
    }
    else if (LorR == L2R)
    {
        cv::imshow("depth L2R", depth_map_HSV);
        cv::imshow("depth L2R nearline", nearline);
    }
    //  std::cout << "block matching time = ";
}

// xmlから外部関数を読み込んでキャリブレーションされた画像を映す関数
void xmlRead()
{
    readXml xml00 = readXml("camera/out_camera_data00.xml");
    readXml xml02 = readXml("camera/out_camera_data02.xml");
    /*
    for (int x = 0; x < xml00.camera_matrix.cols; x++)
    {
        for (int y = 0; y < xml00.camera_matrix.rows; y++)
        {
            std::cout << xml00.camera_matrix.at<double>(y, x) << " ";
        }
        std::cout << std::endl;
    }
    */
    cv::VideoCapture cap(0); // デバイスのオープン
                             // cap.open(0);//こっちでも良い．
                             // capの画像の解像度を変える部分 word CaptureChange
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::VideoCapture cap2(2); // デバイスのオープン
                              // cap.open(0);//こっちでも良い．
                              // capの画像の解像度を変える部分 word CaptureChange
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap2.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap2.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::Mat frame; // 取得したフレーム
    cv::Mat distort;
    cv::Mat matx, maty;

    cv::Mat frame2; // 取得したフレーム
    cv::Mat distort2;

    cv::Mat matx2, maty2;
    cv::initUndistortRectifyMap(xml00.camera_matrix, xml00.distcoeffs, cv::Mat(), xml00.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx, maty);
    cv::initUndistortRectifyMap(xml02.camera_matrix, xml02.distcoeffs, cv::Mat(), xml02.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx2, maty2);

    cv::Mat f1g, f2g;
    int count = 0;
    while (1) // 無限ループ
    {
        cap >> frame;
        cap2 >> frame2;
        int b_time = 0;
        // cv::imshow("win", frame);   // 画像を表示．
        // cv::imshow("win2", frame2); // 画像を表示．
        // f1g.convertTo(f1g, CV_8U);
        // std::cout << f1g << std::endl;
        //  cvt_LBP(frame, distort);
        //  cvt_LBP(frame2, distort2);

        cv::remap(frame, distort, matx, maty, cv::INTER_LINEAR);
        cv::remap(frame2, distort2, matx2, maty2, cv::INTER_LINEAR);

        cv::resize(distort, distort, cv::Size(WIDTH, HEIGHT));
        cv::resize(distort2, distort2, cv::Size(WIDTH, HEIGHT));

        if (BLOCK_MODE == 0)
        {
            cv::cvtColor(distort, distort, cv::COLOR_BGR2GRAY);
            cv::cvtColor(distort2, distort2, cv::COLOR_BGR2GRAY);
        }
        clock_t begin = clock();

        // std::cout << "start block_matching" << std::endl;
        /*
        worker.run([distort2, distort, BLOCK_MODE, R2L, &b_time]()
                   {
        block_Matching(distort, distort2, 3, BLOCK_MODE, R2L);
        b_time++; });

        worker.run([distort2, distort, BLOCK_MODE, L2R, &b_time]()
                   {
        block_Matching(distort2, distort, 3, BLOCK_MODE, L2R);
        b_time++; });

        while (b_time < 2)*/
        ;
#pragma omp section
        // block_Matching(distort, distort2, 3, BLOCK_MODE, R2L);
#pragma omp section
        // block_Matching(distort2, distort, 3, BLOCK_MODE, L2R);
        // std::cout << "end block_matching" << std::endl;
        // clock_t end = clock();
        //  print_elapsed_time(begin, end);

        cv::imshow("a", distort);
        cv::imshow("b", distort2);

        // cv::imwrite(make_spath("images/2023_1128/left", count, tag), distort);
        // cv::imwrite(make_spath("images/2023_1128/right", count, tag), distort2);
        std::cout << "write images " << count << std::endl;
        count++;

        const int key = cv::waitKey(300);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }
    }
    cv::destroyAllWindows();

    return;
}

void subMat()
{
    readXml xml00 = readXml("camera/out_camera_data00.xml");
    readXml xml02 = readXml("camera/out_camera_data02.xml");
    /*
    for (int x = 0; x < xml00.camera_matrix.cols; x++)
    {
        for (int y = 0; y < xml00.camera_matrix.rows; y++)
        {
            std::cout << xml00.camera_matrix.at<double>(y, x) << " ";
        }
        std::cout << std::endl;
    }
    */
    cv::VideoCapture cap(0); // デバイスのオープン
                             // cap.open(0);//こっちでも良い．
                             // capの画像の解像度を変える部分 word CaptureChange
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::VideoCapture cap2(2); // デバイスのオープン
                              // cap.open(0);//こっちでも良い．
                              // capの画像の解像度を変える部分 word CaptureChange
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap2.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap2.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::Mat frame; // 取得したフレーム
    cv::Mat distort;
    cv::Mat matx, maty;

    cv::Mat frame2; // 取得したフレーム
    cv::Mat distort2;

    cv::Mat matx2, maty2;
    cv::initUndistortRectifyMap(xml00.camera_matrix, xml00.distcoeffs, cv::Mat(), xml00.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx, maty);
    cv::initUndistortRectifyMap(xml02.camera_matrix, xml02.distcoeffs, cv::Mat(), xml02.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx2, maty2);

    cv::Mat f1g, f2g;
    // while (1) // 無限ループ
    //{
    cap >> frame;
    cap2 >> frame2;
    // cv::imshow("win", frame);   // 画像を表示．
    // cv::imshow("win2", frame2); // 画像を表示．
    // f1g.convertTo(f1g, CV_8U);
    // std::cout << f1g << std::endl;
    //  cvt_LBP(frame, distort);
    //  cvt_LBP(frame2, distort2);
    //   clock_t begin = clock();
    //    cv::undistort(frame, distort, xml00.camera_matrix, xml00.distcoeffs);
    //    cv::remap(distort, distort, matx, maty, cv::INTER_LANCZOS4);
    //    cv::remap(distort2, distort2, matx2, maty2, cv::INTER_LANCZOS4);
    cv::remap(frame, distort, matx, maty, cv::INTER_LINEAR);
    cv::remap(frame2, distort2, matx2, maty2, cv::INTER_LINEAR);
    cv::cvtColor(distort, f1g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(distort2, f2g, cv::COLOR_BGR2GRAY);
    // clock_t end = clock();
    // print_elapsed_time(begin, end);

    // std::cout << (int)f1g.at<unsigned char>(24, 24) << " " << (int)f2g.at<unsigned char>(24, 24) << " " << abs(f1g.at<unsigned char>(24, 24) - f2g.at<unsigned char>(24, 24)) << std::endl;
    cv::imshow("a", f1g);
    cv::imshow("b", f2g);

    const int key = cv::waitKey(0);
    if (key == 'q' /*113*/) // qボタンが押されたとき
    {
        // break; // whileループから抜ける．
    }
    //}
    cv::destroyAllWindows();

    return;
}

void func(int i)
{
    std::cout << "来たよ！ " << i << std::endl;
}

template <typename F>
void test_run(F &&task)
{
    task;
}

void thread_pool_test()
{
    int sum = 0;
    int time = 0;
    int copy = 0;
    worker_pool<ThreadCount> worker;
    std::vector<BLOCK_MATCHING> s;
    cv::Mat img = cv::imread("images/Test00/000000.jpg", 0);
    cv::Mat img2 = cv::imread("images/Test01/000222.jpg", 0);
    clock_t begin = clock();
    for (int i = 0; i < 100; i++)
    {
        time++;
        copy = i;
        s[i].get_ELEMENTS();
    }
    std::cout << "timeeeeeeeeeeeeeeeeeee " << time << std::endl;
    while (sum < time)
        std::cout << sum << std::endl;
    clock_t end = clock();
    print_elapsed_time(begin, end);
}

void test_img()
{
    cv::Mat left = cv::imread("images/test_img/left05.JPG", BLOCK_MODE);
    cv::Mat right = cv::imread("images/test_img/right05.JPG", BLOCK_MODE);

    cv::resize(left, left, cv::Size(WIDTH, HEIGHT));
    cv::resize(right, right, cv::Size(WIDTH, HEIGHT));

    // cvt_LBP(left, left);
    //  cvt_LBP(right, right);
    //   cv::Canny(left, left, 10, 100);
    //   cv::Canny(right, right, 10, 100);
    //   cv::medianBlur(left, left, 3);
    //   cv::medianBlur(right, right, 3);

    std::cout << "blockmatching" << std::endl;
    clock_t begin = clock();
    /*
    int b_time = 0;
    worker.run([right, left, BLOCK_MODE, R2L, &b_time]()
               {
        block_Matching(right, left, 3, BLOCK_MODE, R2L);
        b_time++; });

    worker.run([right, left, BLOCK_MODE, L2R, &b_time]()
               {
        block_Matching(left, right, 3, BLOCK_MODE, L2R);
        b_time++; });
    while (b_time < 2)
        ;
    */
#pragma omp section
    block_Matching(right, left, 3, BLOCK_MODE, R2L);
#pragma omp section
    block_Matching(left, right, 3, BLOCK_MODE, L2R);

    clock_t end = clock();
    print_elapsed_time(begin, end);

    cv::imshow("left", left);
    cv::imshow("right", right);
    // cv::imshow("c", lbp2);

    const int key = cv::waitKey(0);
    if (key == 'q' /*113*/) // qボタンが押されたとき
    {
        // break; // whileループから抜ける．
    }
}

void test_Mat()
{

    cv::Mat img = cv::imread("images/test_img/left.JPG", 1);
    cv::Mat fimg;
    cv::Canny(img, fimg, 125, 255);
    cv::imwrite("images/test_img/canny.jpg", fimg);
    const int key = cv::waitKey(0);
    if (key == 'q' /*113*/) // qボタンが押されたとき
    {
        // break; // whileループから抜ける．
    }
}

void test_LBP()
{
    cv::Mat img = cv::imread("images/test_img/left05.JPG", BLOCK_MODE);
    cv::Mat lbp;
    cvt_LBP(img, lbp);
    std::ofstream outputfile("logs/LBP_elements/lbp.txt"); // add std::ios::app

    cv::resize(lbp,lbp, cv::Size(WIDTH, HEIGHT));

    cv::Vec3b a = 0;
    cv::Vec3b b = 0;
    cv::Vec3b *src; 
    clock_t begin = clock();
    for (int y = 0; y < lbp.rows; y++)
    {
        for (int x = 0; x < lbp.cols; x++)
        {
            b = lbp.at<cv::Vec3b>(y, x);
        }
    }
    clock_t end = clock();
    print_elapsed_time(begin, end);

    begin = clock();
    for (int y = 0; y < lbp.rows; y++)
    {
        src = lbp.ptr<cv::Vec3b>(y);
        for (int x = 0; x < lbp.cols; x++)
        {
            //outputfile << (int)src[x] << std::endl;
            a = src[x];
        }
    }
    end = clock();
    print_elapsed_time(begin, end);
    outputfile.close();
}
int main()
{
    unsigned int thread_num = std::thread::hardware_concurrency();
    std::cout << "This CPU has " << thread_num << " threads" << std::endl;

    // detective();
    // xmlRead();
    // subMat();
    // thread_pool_test();
    // test_img();
    // test_Mat();
    test_LBP();
    return 0;
}
