// inference.cpp: 定义控制台应用程序的入口点。
//

#include<torch/script.h>
#include <torch/torch.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/parallel/data_parallel.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<cstddef>
#include"args.hxx"
#include"option.hpp"
#include"div2k.hpp"
#include"rdnnet.hpp"

using namespace torch;
using namespace cv;

torch::DeviceType device_type;

struct FloatReader
{
	void operator()(const std::string &name, const std::string &value, std::tuple<double, double> &destination)
	{
		size_t commapos = 0;
		std::get<0>(destination) = std::stod(value, &commapos);
		std::get<1>(destination) = std::stod(std::string(value, commapos + 1));
	}
};

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	cv::Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

//推理
class Inference
{
public:
	Inference() {	}
	~Inference() {	}

public:
	void inference(RDN& model, torch::Device device);
};

void Inference::inference(RDN& model, torch::Device device)
{
	{
		torch::NoGradGuard no_grad_guard;

		cv::Mat targetMat;
		cv::Mat reconstruction_hrMat;
		model.to(device);
		model.train(false);
		model.eval();

		try
		{
			
			cv::Mat img = cv::imread("2023.png", 1);
			torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
			img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous();
			img_tensor = img_tensor.unsqueeze(0);
			img_tensor = img_tensor.toType(torch::kFloat32);
			img_tensor = img_tensor.to(device);
			auto tm_start = std::chrono::system_clock::now();
			auto reconstruction_hr = model.forward(/*input*/img_tensor);
			auto tm_end = std::chrono::system_clock::now();
			printf("It takes %lld msec to finish reconstruction!\n",
				std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
			reconstruction_hr = reconstruction_hr.squeeze(0);
			reconstruction_hr = reconstruction_hr.mul(1).clamp(0, 255).round().div(1);
			torch::Tensor image_rec = reconstruction_hr.permute({ 1, 2, 0 }).contiguous();
			image_rec = image_rec.toType(torch::kByte);
			image_rec = image_rec.to(device);
			reconstruction_hrMat = ToCvImage(image_rec);
		}
		catch (const c10::Error& e)
		{
			std::cerr << e.msg();
			//return -1;
		}
		cv::imshow("rec", reconstruction_hrMat);
		cv::waitKey(5000);
	}
}

COptionMap<int> COptInt;
COptionMap<bool> COptBool;
COptionMap<std::string> COptString;
COptionMap<double> COptDouble;
COptionMap<std::tuple<double, double>> COptTuple;

int main(int argc, char **argv)
{
	if (torch::cuda::is_available())
	{
		printf("torch::cuda::is_available\n");
		device_type = torch::kCUDA;
	}
	else
	{
		printf("cpu is_available");
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	srand(time(NULL));

	args::ArgumentParser parser("This is a super resolution reconstruction using rdn.", "This goes after the options.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	//# Hardware specifications
	args::ValueFlag<int> n_threads(parser, "n_threads", "number of threads for data loading", { "n_threads" }, 1);
	args::ValueFlag<bool> cpu(parser, "cpu", "use cpu only", { "cpu" }, 0);
	args::ValueFlag<int> n_GPUs(parser, "n_GPUs", "number of GPUs", { "n_GPUs" }, 1);
	args::ValueFlag<int> seed(parser, "seed", "random seed", { "seed" }, 1);

	//# Data specifications
	args::ValueFlag<std::string> dir_data(parser, "dir_data", "dataset directory", { "dir_data" }, ".\\DIV2K");
	args::ValueFlag<std::string> dir_demo(parser, "dir_demo", "demo image directory", { "dir_demo" }, "..\\test");
	args::ValueFlag<std::string> data_train(parser, "data_train", "train dataset name", { "data_train" }, "DIV2K");
	args::ValueFlag<std::string> data_test(parser, "data_test", "test dataset name", { "data_test" }, "DIV2K");
	args::ValueFlag<std::string> data_range(parser, "data_range", "train/test data range", { "data_range" }, "1-879/870-879");
	args::ValueFlag<std::string> ext(parser, "ext", "dataset file extension", { "ext" }, "sep");
	args::ValueFlag<std::string> scale(parser, "scale", "super resolution scale", { "scale" }, "2");
	args::ValueFlag<int> patch_size(parser, "patch_size", "output patch size", { "patch_size" }, 16);
	args::ValueFlag<int> rgb_range(parser, "rgb_range", "maximum value of RGB", { "rgb_range" }, 255);
	args::ValueFlag<int> n_colors(parser, "n_colors", "number of color channels to use", { "n_colors" }, 3);
	args::ValueFlag<bool> chop(parser, "chop", "enable memory-efficient forward", { "chop" }, 0);
	args::ValueFlag<bool> no_augment(parser, "no_augment", "do not use data augmentation", { "no_augment" }, 0);

	//# Model specifications
	args::ValueFlag<std::string> model(parser, "model", "model name", { "model" }, "RDN");
	args::ValueFlag<std::string> act(parser, "act", "activation function", { "act" }, "relu");
	args::ValueFlag<std::string> pre_train(parser, "pre_train", "pre-trained model directory", { "pre_train" }, "");
	args::ValueFlag<std::string> extend(parser, "extend", "pre-trained model directory", { "extend" }, ".");
	args::ValueFlag<int> n_resblocks(parser, "n_resblocks", "number of residual blocks", { "n_resblocks" }, 16);
	args::ValueFlag<int> n_feats(parser, "n_feats", "number of feature maps", { "n_feats" }, 64);
	args::ValueFlag<float> res_scale(parser, "res_scale", "residual scaling", { "res_scale" }, 1);
	args::ValueFlag<bool> shift_mean(parser, "shift_mean", "subtract pixel mean from the input", { "shift_mean" }, 1);
	args::ValueFlag<bool> dilation(parser, "dilation", "use dilated convolution", { "dilation" }, 1);
	args::ValueFlag<std::string> precision(parser, "precision", "FP precision for test (single | half)", { "precision" }, "single");

	//# Option for Residual dense network(RDN)
	args::ValueFlag<int> G0(parser, "G0", "default number of filters. (Use in RDN)", { "G0" }, 64);
	args::ValueFlag<int> RDNkSize(parser, "RDNkSize", "default kernel size. (Use in RDN)", { "RDNkSize" }, 3);
	args::ValueFlag<std::string> RDNconfig(parser, "RDNconfig", "parameters config of RDN. (Use in RDN)", { "RDNconfig" }, "B");

	//# Option for Residual channel attention network(RCAN)
	args::ValueFlag<int> n_resgroups(parser, "n_resgroups", "number of residual groups", { "n_resgroups" }, 10);
	args::ValueFlag<int> reduction(parser, "reduction", "number of feature maps reduction", { "reduction" }, 16);

	//# Training specifications
	args::ValueFlag<bool> reset(parser, "reset", "reset the training", { "reset" }, 1);
	args::ValueFlag<int> test_every(parser, "test_every", "do test per every N batches", { "test_every" }, 1000);
	args::ValueFlag<int> epochs(parser, "epochs", "number of epochs to train", { "epochs" }, 15000);
	args::ValueFlag<int> batch_size(parser, "batch_size", "input batch size for training", { "batch_size" }, 16);
	args::ValueFlag<int> split_batch(parser, "split_batch", "split the batch into smaller chunks", { "split_batch" }, 1);
	args::ValueFlag<bool> self_ensemble(parser, "self_ensemble", "use self-ensemble method for test", { "self_ensemble" }, 0);
	args::ValueFlag<bool> test_only(parser, "test_only", "set this option to test the model", { "test_only" }, 0);
	args::ValueFlag<int> gan_k(parser, "gan_k", "k value for adversarial loss", { "gan_k" }, 1);

	//# Optimization specifications
	args::ValueFlag<double> lr(parser, "lr", "learning rate", { "lr" }, 1e-4);
	args::ValueFlag<std::string> decay(parser, "decay", "learning rate decay type", { "decay" }, "200");
	args::ValueFlag<float> gamma(parser, "gamma", "learning rate decay factor for step decay", { "gamma" }, 0.5);
	args::ValueFlag<std::string> optimizer(parser, "optimizer", "optimizer to use (SGD | ADAM | RMSprop)", { "optimizer" }, "ADAM");
	args::ValueFlag<double> momentum(parser, "momentum", "SGD momentum", { "momentum" }, 0.9);
	args::ValueFlag<std::tuple<double, double>, FloatReader> betas(parser, "betas", "ADAM beta", { "betas" }, std::tuple<double, double>(0.9, 0.999));
	args::ValueFlag<double> epsilon(parser, "epsilon", "ADAM epsilon for numerical stability", { "epsilon" }, 1e-8);
	args::ValueFlag<double> weight_decay(parser, "weight_decay", "weight decay", { "weight_decay" }, 0);
	args::ValueFlag<double> gclip(parser, "gclip", "gradient clipping threshold (0 = no clipping)", { "gclip" }, 0);

	//# Loss specifications
	args::ValueFlag<std::string> loss(parser, "loss", "loss function configuration", { "loss" }, "1*L1");
	args::ValueFlag<double> skip_threshold(parser, "skip_threshold", "skipping batch that has large error", { "skip_threshold" }, 1e8);

	//# Log specifications
	args::ValueFlag<std::string> save(parser, "save", "file name to save", { "save" }, "test");
	args::ValueFlag<std::string> load(parser, "load", "file name to load", { "load" }, "");
	args::ValueFlag<int> resume(parser, "resume", "resume from specific checkpoint", { "resume" }, 0);
	args::ValueFlag<bool> save_models(parser, "save_models", "save all intermediate models", { "save_models" }, 0);
	args::ValueFlag<int> print_every(parser, "print_every", "how many batches to wait before logging training status", { "print_every" }, 100);
	args::ValueFlag<bool> save_results(parser, "save_results", "save output results", { "save_results" }, 0);
	args::ValueFlag<bool> save_gt(parser, "save_gt", "save low-resolution and high-resolution images together", { "save_gt" }, 0);

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}
	catch (args::ValidationError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	//{ std::cout << "n_threads: " << args::get(n_threads) << std::endl; }
	COptInt.InsertOptMap("n_threads", (int)args::get(n_threads));
	COptInt.InsertOptMap("rgb_range", (int)args::get(rgb_range));
	COptInt.InsertOptMap("n_colors", (int)args::get(n_colors));
	COptInt.InsertOptMap("G0", (int)args::get(G0));
	COptInt.InsertOptMap("RDNkSize", (int)args::get(RDNkSize));
	COptInt.InsertOptMap("epochs", (int)args::get(epochs));

	//{ std::cout << "cpu: " << args::get(cpu) << std::endl; }
	COptBool.InsertOptMap("cpu", (bool)args::get(cpu));
	COptBool.InsertOptMap("no_augment", (bool)args::get(no_augment));

	Inference		I;
	RDN			rdn(device_type);
	torch::serialize::InputArchive archive;
	archive.load_from("..\\retModel\\rdn.pt");

	rdn.load(archive);

	I.inference(rdn, device);

    return 0;
}

