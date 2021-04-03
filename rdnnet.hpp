#include<torch/script.h>
#include <torch/torch.h>
//#include <torch/nn/pimpl.h>
//#include <torch/nn/parallel/data_parallel.h>
#include"option.hpp"

extern torch::DeviceType device_type;
extern	COptionMap<int> COptInt;
extern	COptionMap<bool> COptBool;

struct RDB_Conv : public torch::nn::Module
{
	RDB_Conv(int inChannels, int growRate, int kSize = 3)
	{
		Cin = inChannels;
		G = growRate;

		torch::nn::Sequential conv(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(Cin, G, kSize).padding((kSize - 1) / 2).stride(1)),
			torch::nn::ReLU());

		RDB_CONV = register_module("RDB_Conv", conv.ptr());
	}

	~RDB_Conv()
	{

	}

	torch::Tensor forward(torch::Tensor input)
	{
		torch::Tensor out = RDB_CONV->forward(input);
		return torch::cat({ input, out }, 1);
	}

	int Cin = 0, G = 0;
	torch::nn::Sequential RDB_CONV;
};

struct RDB : public torch::nn::Module
{
	RDB(int growRate0, int growRate, int nConvLayers, int kSize = 3)
	{
		G0 = growRate0;
		G = growRate;
		C = nConvLayers;

		
		for (int i = 0; i < C; i++)
			rdb->push_back(RDB_Conv(G0 + i * G, G));

		RDB_SEQ = register_module("RDB", rdb.ptr());

		//Local Feature Fusion
		LFF = register_module("LFF", torch::nn::Conv2d(torch::nn::Conv2dOptions(G0 + C * G, G0, 1).padding(0).stride(1)));
	}

	~RDB()
	{

	}

	torch::Tensor forward(torch::Tensor input)
	{
		auto x = LFF->forward(RDB_SEQ->forward(input));
		auto ret = x.add(input);
		return ret;
	}

	int G0 = 0, G = 0, C = 0;
	torch::nn::Sequential rdb;
	torch::nn::Sequential RDB_SEQ;
	torch::nn::Conv2d	LFF{ nullptr };
};
struct RDN : public torch::nn::Module
{
	RDN(torch::Device device):Device(device)
	{
		int n_colors = 0, G0 = 0, RDNkSize = 0;
		COptInt.GetOptMap("n_colors", n_colors);
		COptInt.GetOptMap("G0", G0);
		COptInt.GetOptMap("RDNkSize", RDNkSize);
		kSize = RDNkSize;
		printf("n_colors : %d  G0 : %d  RDNkSize : %d kSize : %d padding : %d \n", n_colors, G0, RDNkSize, kSize, (kSize - 1) / 2);
		//Expected 4 - dimensional input for 4 - dimensional weight[64, 3, 3, 3], but got 3 - dimensional input of size[3, 8, 8] instead
		SFENet1 = register_module("SFENet1", torch::nn::Conv2d(torch::nn::Conv2dOptions(n_colors, G0, { RDNkSize, RDNkSize }).padding((kSize - 1) / 2).stride(1)));
		SFENet2 = register_module("SFENet2", torch::nn::Conv2d(torch::nn::Conv2dOptions(G0, G0, kSize).padding((kSize - 1) / 2).stride(1)));

		for(int i = 0; i < 16; i ++)
			RDBs->push_back(RDB(G0, 64, 8));
		RDBsModel = register_module("RDBs", RDBs.ptr());

		torch::nn::Sequential GFF(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(16 * 64, 64, 1).padding(0).stride(1)), 
			torch::nn::Conv2d(torch::nn::Conv2dOptions(G0, G0, kSize).padding((kSize - 1) / 2).stride(1)));

		GFFModel = register_module("GFF", GFF.ptr());
		
		torch::nn::Sequential UPNet(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(G0, 64 * 2 * 2, kSize).padding((kSize - 1) / 2).stride(1)),
			torch::nn::PixelShuffle(2),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(64, n_colors, kSize).padding((kSize - 1) / 2).stride(1))
		);
		UPNetModel = register_module("UPNet", UPNet.ptr());
	}

	~RDN()
	{

	}

	torch::Tensor forward(torch::Tensor input)
	{
		input = input.to(Device);
		f__1 = SFENet1(input);
		x = SFENet2(f__1);
		for (int i = 0; i < 16; i++)
		{
			RDB m = RDBsModel->at<RDB>(i);
			x = m.forward(x);
			RDBs_out[i] = x;
		}
		
		x = GFFModel->forward(torch::cat(RDBs_out, 1));
		auto xx = x.add(f__1);


		return UPNetModel->forward(xx);
	}

	int kSize = 0;
	torch::Tensor f__1;
	torch::Tensor x;
	torch::Tensor RDBs_out[16];
	torch::nn::Conv2d	SFENet1{ nullptr };
	torch::nn::Conv2d	SFENet2{ nullptr };
	torch::nn::ModuleList RDBs;
	torch::nn::ModuleList RDBsModel;
	torch::nn::Sequential GFFModel;
	torch::nn::Sequential UPNetModel;
	torch::Device Device;
};