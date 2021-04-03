#include"option.hpp"
#include"srdata.hpp"

#include <ctime>
#include <cstdlib>

#define N  999

extern	COptionMap<int> COptInt;
extern	COptionMap<bool> COptBool;
//When the tensor was created using a c10::kByte for the cast we need to use uchar and not char or uint,
//etc . so in order to get this fixed I only had to use uchar instead of int:
//Side note : In case you created your Tensor with any other type,
//make sure to use Tensor::totype() effectively and convert to the proper type before hand.That is before I feed this tensor to my network,
//e.g I convert it to KFloat and then carry on!its an obvious point that may very well be neglected and cost you hours of debugging!
//因为使用了这个转换,在把tensor数据送到网络之前千万记得把tensor类型改成float
auto ToCvImage(at::Tensor tensor)
{
	int width = tensor.sizes()[1];
	int height = tensor.sizes()[0];
	try
	{
		cv::Mat output_mat(cv::Size{ width, height }, CV_8UC3, tensor.data_ptr<uchar>());
		return output_mat.clone();
	}
	catch (const c10::Error& e)
	{
		std::cout << "an error has occured : " << e.msg() << std::endl;
	}
	return cv::Mat(height, width, CV_8UC3);
}
typedef struct stuArgs4DIV2K
{
	std::string		_str_Data_range;
	STUARGS4SRDATA	_stuArgs4SRData;

}STUARGS4DIV2K, *PSTUARGS4DIV2K;

/* Convert and Load image to tensor from location argument */
torch::Tensor read_data(std::string location)
{
	cv::Mat img = cv::imread(location, 1);
	// Convert image to tensor
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous(); // Channels x Height x Width
	// Read Data here
	// Return tensor form of the image
	return img_tensor.clone();
}

/* Converts label to tensor type in the integer argument */

//根据数组创建一个tensor
//double array[] = { 1, 2, 3, 4, 5 };
//auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 1);
//torch::Tensor tharray = torch::from_blob(array, { 5 }, options);
torch::Tensor read_label(std::string location)
{
	cv::Mat img = cv::imread(location, 1);
	// Convert image to tensor
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous(); // Channels x Height x Width
	// Read Data here
	// Return tensor form of the image
	return img_tensor.clone();
}

///* Loads images to tensor type in the string argument */
//vector<torch::Tensor> process_images(vector<string> list_images) {
//	cout << "Reading Images..." << endl;
//	// Return vector of Tensor form of all the images
//	vector<torch::Tensor> states;
//	for (std::vector<string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
//		torch::Tensor img = read_data(*it);
//		states.push_back(img);
//	}
//	return states;
//}

/* Loads images to tensor type in the string argument */
vector<torch::Tensor> process_images(fileNameList_t list_images) {
	cout << "Reading Images..." << endl;
	// Return vector of Tensor form of all the images
	vector<torch::Tensor> states;
	for (fileNameList_t::iterator it = list_images.begin(); it != list_images.end(); ++it) {
		torch::Tensor img = read_data(it->strFilePath);
		states.push_back(img);
	}
	return states;
}

/* Loads labels to tensor type in the string argument */
//vector<torch::Tensor> process_labels(vector<string> list_labels) {
//	cout << "Reading Labels..." << endl;
//	// Return vector of Tensor form of all the labels
//	vector<torch::Tensor> labels;
//	for (std::vector<string>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
//		torch::Tensor label = read_label(*it);
//		labels.push_back(label);
//	}
//	return labels;
//}

/* Loads labels to tensor type in the string argument */
vector<torch::Tensor> process_labels(fileNameList_t list_labels) {
	cout << "Reading Labels..." << endl;
	// Return vector of Tensor form of all the labels
	vector<torch::Tensor> labels;
	for (fileNameList_t::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
		torch::Tensor label = read_label(it->strFilePath);
		labels.push_back(label);
	}
	return labels;
}

class DIV2K :public SRData
{
public:
	DIV2K(std::string strName = "DIV2K", bool bTrain = true, bool bBenchmark = false) :SRData("", true, false) { train_ = bTrain; }
	~DIV2K() {}

	STUARGS4DIV2K stuArgs4DIV2K;

	void DIV2KFunc_Init(std::string strDirdata, std::string strExt);

	void DIV2KFunc_ProcessImage(fileNameList_t, fileNameList_t);

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;

private:
	vector<torch::Tensor> lrimages_, hrimages_;

	bool	train_;

	//将一张图片截取出一小块,低分辨*2对应高分辨的区域
	torch::data::Example<> _get_patch(torch::Tensor, torch::Tensor, int nPatch_size = 64, int nScale = 2, bool bMulti = false, bool bInput_large = false);
};

torch::data::Example<> DIV2K::get(size_t index)
{
	torch::Tensor sample_img = lrimages_.at(index);
	torch::Tensor sample_label = hrimages_.at(index);

	torch::Tensor ret_img;
	torch::Tensor ret_label;

	int nBatch_Size = 0;
	SRDataFunc_GetBatchSize(nBatch_Size);
	std::string strScale;
	SRDataFunc_GetScale(strScale);
	if (train_)	//如果是训练状态则对lr截取8*8图片区域和对应scale hr的16*16以及进行数据增强
	{
		return _get_patch(sample_img, sample_label, nBatch_Size, atoi(strScale.c_str()), false, false);
	}
	else		//如果非训练而是测试的,则直接返回lr和hr全尺寸图片
	{
		return { sample_img.clone(), sample_label.clone() };
	}
}

torch::optional<size_t> DIV2K::size() const
{
	return hrimages_.size();
}

// 对图片进行裁剪,根据一些参数,例如本次运行scale为2则裁剪成lr 为 8 * 8 hr 为 16 * 16 两个区域是准确的对应
torch::data::Example<> DIV2K::_get_patch(torch::Tensor lrImage, torch::Tensor hrImage, int nPatch_size, int nScale, bool bMulti, bool bInput_large)
{
	torch::Tensor sample_img;
	torch::Tensor sample_label;

	int iw = lrImage.sizes().at(1);
	int ih = lrImage.sizes().at(2);
	int p = 0, tp = 0, ip = 0;
	int ix = 0, iy = 0;
	int tx = 0, ty = 0;

	if (!bInput_large)
	{
		if (bMulti)
			p = nScale;
		else
			p = 1;
		tp = p * nPatch_size;
		ip = tp / nScale;
	}
	else
	{
		tp = nPatch_size;
		ip = nPatch_size;
	}

	//一般性：rand() % (b-a+1)+ a ;    就表示  a~b 之间的一个随机整数。
	//8 - iw,8 - ih
	ix = rand() % ((iw - ip) - ip + 1) + ip;
	iy = rand() % ((ih - ip) - ip + 1) + ip;

	if (!bInput_large)
	{
		tx = nScale * ix;
		ty = nScale * iy;
	}
	else
	{
		tx = ix;
		ty = iy;
	}

	bool	bNo_augment = false;
	bool	bhflip = false;		//水平翻转
	bool	bvflip = false;		//上下翻转
	bool	brot90 = false;
	COptBool.GetOptMap("no_augment", bNo_augment);

	int nrgb_range = 0;
	COptInt.GetOptMap("rgb_range", nrgb_range);

	if (!bNo_augment)		//是否进行数据增强
	{
		if ((rand() % (N + 1) / (float)(N + 1)) < 0.5)
		{
			bhflip = true;
		}
		if ((rand() % (N + 1) / (float)(N + 1)) < 0.5)
		{
			bvflip = true;
		}
		if ((rand() % (N + 1) / (float)(N + 1)) < 0.5)
		{
			brot90 = true;
		}
	}

	torch::Tensor imagelr_tensor = lrImage.permute({ 1, 2, 0 }).contiguous();
	cv::Mat lrMat = ToCvImage(imagelr_tensor);
	Mat copylrMat(lrMat.rows, lrMat.cols, CV_8UC3);
	lrMat(Rect(iy, ix, ip, ip)).copyTo(copylrMat);
	if (bhflip)
		cv::flip(copylrMat, copylrMat,1);
	if (bvflip)
		cv::flip(copylrMat, copylrMat, 0);
	if (brot90)
		cv::transpose(copylrMat, copylrMat);
	sample_img = torch::from_blob(copylrMat.data, { copylrMat.rows, copylrMat.cols, 3 }, torch::kByte);
	sample_img = sample_img.permute({ 2, 0, 1 }).contiguous();
	sample_img.mul_(nrgb_range / 255);

	torch::Tensor imagehr_tensor = hrImage.permute({ 1, 2, 0 }).contiguous();
	cv::Mat hrMat = ToCvImage(imagehr_tensor);
	Mat copyhrMat(hrMat.rows, hrMat.cols, CV_8UC3);
	hrMat(Rect(ty, tx, tp, tp)).copyTo(copyhrMat);
	if (bhflip)
		cv::flip(copyhrMat, copyhrMat, 1);
	if (bvflip)
		cv::flip(copyhrMat, copyhrMat, 0);
	if (brot90)
		cv::transpose(copyhrMat, copyhrMat);
	sample_label = torch::from_blob(copyhrMat.data, { copyhrMat.rows, copyhrMat.cols, 3 }, torch::kByte);
	sample_label = sample_label.permute({ 2, 0, 1 }).contiguous();
	sample_label.mul_(nrgb_range / 255);

	return { sample_img.clone() , sample_label.clone() };
}

void DIV2K::DIV2KFunc_Init(std::string strDirdata, std::string strExt)
{
	SRDataFunc_SetInputLarge(stuArgs4DIV2K._stuArgs4SRData._strModule);
	SRDataFunc_SetScale(stuArgs4DIV2K._stuArgs4SRData._strScale);
	SRDataFunc_SetBatchSize(stuArgs4DIV2K._stuArgs4SRData._patch_size);
	SRDataFunc_SetTestEvery(stuArgs4DIV2K._stuArgs4SRData._test_every);
	SRDataFunc_Init(strDirdata, strExt);
}

void DIV2K::DIV2KFunc_ProcessImage(fileNameList_t listLrFileName, fileNameList_t listHrFileName)
{
	lrimages_ = process_images(listLrFileName);
	hrimages_ = process_labels(listHrFileName);

	//std::random_shuffle(lrimages_.begin(), lrimages_.end());
}