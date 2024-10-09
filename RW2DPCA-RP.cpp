#include<opencv2/opencv.hpp>
#include<iostream>
#include<stdio.h>
#include"time.h"
#include<windows.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib")
#include <stdio.h>

#include <math.h>
using namespace std;
using namespace cv;

// 函数：显示程序使用的内存信息
void showMemoryInfo(void)
{
	HANDLE handle = GetCurrentProcess();
	PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(handle, &pmc, sizeof(pmc));
	cout << "内存使用：" << pmc.WorkingSetSize / 1000 << "K/" << pmc.PeakWorkingSetSize / 1000 << "K + " << pmc.PagefileUsage / 1000 << "K/" << pmc.PeakPagefileUsage / 1000 << "K" << endl;
}

// 函数：计算矩阵 A 的元素平方和的平方根
double sumF(Mat A)
{
	double s = 0.0;
	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < A.cols; j++)
		{
			s += A.at<float>(i, j) * A.at<float>(i, j);
		}
	}
	s = sqrt(s);
	return s;
}

void main()
{
	// 开始计时
	clock_t start, finish;
	double duration;
	start = clock();

	// 计算内存使用情况
	showMemoryInfo();
	cout << "回收所有可回收的内存" << endl;
	EmptyWorkingSet(GetCurrentProcess());
	showMemoryInfo();
	cout << "开始动态分配内存" << endl;

	// 定义变量
#define eigenNum  30 // 特征向量数量（可调整）

	int sampleNum = 160;        // 样本数量（可调整）
	//int originrows = 112, origincols = 92;//resize
	int nrows = 200, ncols = 50; // 样本大小（可调整）
	float  s = 2; // 可调整
	int classNum = 8; // 训练集类别数
	int perclassNum = 20; // 每类训练集图片的数量
	int max_iteration = 40; // 迭代最大次数
	Mat meanSample = Mat::zeros(nrows, ncols, CV_32FC1);    // 样本的平均值
	Mat totalSample = Mat::zeros(nrows, ncols, CV_32FC1);   // 样本的总和
	Mat oneSample(nrows, ncols, CV_32FC1);
	// Mat sizeSample = Mat::zeros(nrows, ncols, CV_32FC1);
	int ConstrainNum = 1; // 约束总块数，要保证 nrows / ConstrainNum 为整数
	int computerNum = sampleNum * nrows / ConstrainNum; // 操作的约束后的块数
	Mat ag = Mat::zeros(ConstrainNum, ncols, CV_32FC1);

	// 初始化 W
	Mat U, S, VT;
	vector<Mat> samples;        // 存储样本
	vector<Mat> cosamples;        // 存储约束样本
	// 加载样本并计算总样本
	for (int i = 0; i < sampleNum; i++)
	{
		string imagePath = "E:\\random_noise_databases\\ETH-80_original后做数据集\\ETH_0.4_0.10-0.30_100x100\\block200x50\\train12\\" + to_string(static_cast<long long>(i)) + ".png"; // 图像路径
		Mat srcImage = imread(imagePath, 0);
		srcImage.clone().convertTo(oneSample, CV_32FC1, 1, 0);
		// resize(oneSample, sizeSample, sizeSample.size());
		totalSample += oneSample;
		samples.push_back(oneSample.clone());
	}

	//load the samples and compute the total samples    按类别分的数据集
	/*for (int q = 1; q <= classNum; q++) {
		for (int i = 0; i < perclassNum; i++)
		{
			string imagePath = "E:\\random_noise_databases\\NEC后做数据集\\NEC_0.4_0.10-0.30_96X116\\train1\\" + to_string(static_cast<long long>(q)) + "\\" + to_string(static_cast<long long>(i)) + ".jpg";//imagepath to_string(static_cast<long long>(i))
			Mat srcImage = imread(imagePath, 0);
			srcImage.clone().convertTo(oneSample, CV_32FC1, 1, 0);
			totalSample += oneSample;
			samples.push_back(oneSample.clone());
		}
	}
	*/
	//compute the mean sample
	meanSample = totalSample / double(sampleNum);
	//make constrain sample
	for (int i = 0; i < sampleNum; i++)
	{
		for (int j = 0; j < nrows / ConstrainNum; j++) {
			Mat J = samples[i] - meanSample;
			Mat a = J.rowRange(j*ConstrainNum, (j + 1)*ConstrainNum);
			cosamples.push_back(a.clone());
		}
	}
	for (int K = 5; K <= eigenNum; K = K + 5)
	{
		cout << K << endl;
		int iteration = 0;
		Mat I = Mat::eye(ncols, K, CV_32FC1);
		float d = 0;
		Mat G = Mat::zeros(ncols, ncols, CV_32FC(1));
		Mat H = Mat::zeros(ncols, K, CV_32FC(1));
		Mat w = Mat::eye(ncols, K, CV_32FC(1));//注意初始赋值细节，否则会得出0矩阵错误结果
		w = w / 20;  //让w主对角线元素变得更小，利于收敛
		Mat ws = Mat::zeros(ncols, K, CV_32FC(1));
		Mat store_redusial = Mat::zeros(max_iteration, 1, CV_32FC1);
		do {

			iteration = iteration + 1;
			//float s = 1;
			double E = 0, C = 0, F = 0, objective_function = 0;
			H = 0; double d = 0;
			double gamu = 0.000001;
			w.copyTo(ws);//保存w的副本
			for (int i = 0; i < computerNum; i++)
			{
				ag = cosamples[i];
				C = pow(sumF(ag), s);
				//cout << "C=" << C << endl;

				F = sumF(ag*w);
				//E = pow(E, s);
				//for (int j = 0; j < BlockNum; j++) {C = sumL2(ag.row(j));d += pow(C, p );}

				//cout << "d=" << d << endl;
				//a的b次方
				E = sumF(ag - ag * w*w.t());
				d = C * F*E + gamu;// 避免为零
				H += ag.t()*ag*w / d;
				//objective_function += E * F / C;

			}
			

			SVD::compute(H, S, U, VT, SVD::FULL_UV);
			w = U * I*VT;
			//cout << "objective_function="<< objective_function << endl;
			//store_redusial.at<float>(iteration-1,0) = float(sumF(w - ws));
			if (iteration >= max_iteration) {
				cout << "该收敛条件下，K=" << K << "无法进行，需要调整收敛条件,以上为w-ws的F范数残差" << endl;
				//cout << store_redusial << endl;
				break;//跳出while循环，直接保存最后一次迭代的结果

			}
			cout << sumF(w - ws) << endl;
		} while (sumF(w - ws) / sumF(w) > 0.008); //收敛条件

		CvMat vecs = w;
		char store[256]; sprintf_s(store, "E:\\所有实验test test对比\\BPCA\\Height_diffnorms_2DPCA\\ETH200x50\\12\\hebing%d.txt", K);
		cvSave(store, &vecs);

		CvMat agSample = meanSample;
		cvSave("E:\\所有实验test test对比\\BPCA\\Height_diffnorms_2DPCA\\ETH200x50\\12\\meanSample1.txt", &agSample);
		cout << "iteration=" << iteration << endl;
		//计算训练特征矩阵
	/*	Mat trainDataMat = Mat::zeros(nrows, eigenNum, CV_32FC1);;
		for (int i = 0; i < sampleNum; i++)
		{

			Mat srcSample = samples[i] - meanSample;//中心化处理
			trainDataMat = srcSample * w;
			cout << "image_id=" << i << endl;
			for (int ii = 0; ii < eigenNum; ii++) {
				for (int j = 0; j < nrows; j++) {
					cout << trainDataMat.at<float>(j, ii) << endl;
				}cout << endl;
			}*/
			/*CvMat storeMat;
			storeMat = trainDataMat;
			char store[256]; sprintf_s(store, "E:\\random_noise_databases\\可视化数据\\生成的降维子空间投影\\GHEIGHT\\s=2\\resize\\noise\\2维投影图片%d.txt", i);
			cvSave(store, &storeMat);*/

			//}
	}

	//using memory
	showMemoryInfo();


	//finish timing
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "训练过程结束,共耗时：" << duration << "秒" << endl;

	waitKey(0);

}
