#include<torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include"option.hpp"
#include"common.hpp"

#include <iostream>
#include <cstring>        // for strcpy(), strcat()
#include <io.h>
#include <stdio.h>

using namespace std;
using namespace torch;
using namespace cv;

typedef struct stuArgs4SRData
{
	std::string		_strModule;
	std::string		_strScale;
	std::string		_dir_data;
	//std::string		_ext;
	std::tuple<std::string, std::string> _ext;
	int				_patch_size;
	int				_test_every;
}STUARGS4SRDATA, *PSTUARGS4SRDATA;

typedef struct stuFileInfoList
{
	std::string		strFilePath;
	std::string		strFileName;
}STUFILEINFOLIST, *PSTUFILEINFOLIST;

typedef std::list< STUFILEINFOLIST > fileNameList_t;

void append(fileNameList_t &List, STUFILEINFOLIST &info);
void for_each(fileNameList_t &List);

void append(fileNameList_t &List, STUFILEINFOLIST &info)
{
	List.push_back(info);
}

void for_each(fileNameList_t &List)
{
	fileNameList_t::iterator iter;
	for (iter = List.begin(); iter != List.end(); iter++)
	{
		std::cout << iter->strFilePath << std::endl;
		//std::cout << iter->strFileName << std::endl;
	}
}

class SRData : public torch::data::Dataset<SRData>
{
public:
	explicit SRData(std::string strName = "", bool train = true, bool benchmark = false):
		strName_(strName),
		bTrain_(train),
		bDo_eval_(true),
		bBenchmark_(benchmark),
		bInput_large_(false),
		strScale_("2")
	{
		if (bTrain_)
			strSplit_ = "train";
		else
			strSplit_ = "test";
	};

	STUARGS4SRDATA		stuArgs4SRData;

	//要在参数结构体赋值之后才有正确值的变量使用单独函数进行赋值和取值
	void SRDataFunc_SetInputLarge(std::string strModule) { if (0 == strModule.compare("VDSR")) bInput_large_ = true; else  bInput_large_ = false; }
	bool SRDataFunc_GetInputLarge() { return bInput_large_; }
	void SRDataFunc_SetScale(std::string strScale) { strScale_ = strScale; }
	void SRDataFunc_GetScale(std::string & strScale) { strScale = strScale_; }
	void SRDataFunc_SetBatchSize(int nBatch_size) { batch_size_ = nBatch_size; }
	void SRDataFunc_GetBatchSize(int & nBatch_size) { nBatch_size = batch_size_; }
	void SRDataFunc_SetTestEvery(int nTest_every) { test_every_ = nTest_every; }
	void SRDataFunc_GetTestEvery(int & nTest_every) { nTest_every = test_every_; }

	void SRDataFunc_Init(std::string strDirdata, std::string strExt);

	//把高分辨率图片和低分辨率图片对应完整路径全部放这两个list
	//其中Hr为标签,Lr为重建前图片
	fileNameList_t	listHrFileName_;
	fileNameList_t	listLrFileName_;
private:

	std::string strName_;
	bool bTrain_;
	std::string strSplit_;
	bool bDo_eval_;
	bool bBenchmark_;
	bool bInput_large_;
	std::string strScale_;
	int batch_size_;
	int test_every_;

	std::string strApath_;
	std::string	strDir_hr_;
	std::string strDir_lr_;
	std::string strExt1_;
	std::string strExt2_;

private:
	void _set_filesystem_(std::string dir_data);
	void _scan_(std::string strDirdata, std::string strExt, fileNameList_t& _t);

	void _listFiles_(const char * dir, fileNameList_t&);
};

void SRData::_set_filesystem_(std::string strDir_data)
{
	strApath_ = strDir_data;
	strDir_hr_ = strApath_ + "_" + "HR";
	strDir_lr_ = strApath_ + "_" + "LR_bicubic";

	if (bInput_large_)
		strDir_lr_ += 'L';

	//cout << "apath : " << strApath_ << endl;
	//cout << "dir_hr : " << strDir_hr_ << endl;
	//cout << "dir_lr : " << strDir_lr_ << endl;
}

void SRData::SRDataFunc_Init(std::string strDirdata, std::string strExt)
{
	_set_filesystem_(strDirdata);
	_scan_(strDir_hr_, strExt, listHrFileName_);
	std::string strlr = strDir_lr_ + "\\" + "X" + strScale_;
	_scan_(strlr, strExt, listLrFileName_);
}

void SRData::_scan_(std::string strDirdata, std::string strExt, fileNameList_t& _t)
{
	strExt1_ = "\\*." + strExt;
	_listFiles_(strDirdata.c_str(), _t);
	//cout << strDirdata << endl;
	//for_each(_t);
}

void SRData::_listFiles_(const char * dir, fileNameList_t& _t)
{
	std::string		strDir = dir;
	char dirNew[200];
	strcpy_s(dirNew, 200, dir);
	strcat_s(dirNew, 200, strExt1_.c_str());    // 在目录后面加上"\\*.*"进行第一次搜索
	//cout <<"xxxx"<< dirNew << endl;

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dirNew, &findData);
	if (handle == -1)        // 检查是否成功
		return;

	do
	{
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;

			//cout << findData.name << "\t<dir>\n";

			// 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
			strcpy_s(dirNew, 200, dir);
			strcat_s(dirNew, 200, "\\");
			strcat_s(dirNew, 200, findData.name);

			_listFiles_(dirNew, _t);
		}
		else
		{
			STUFILEINFOLIST stuFileInfo;
			stuFileInfo.strFilePath = strDir + "\\" + findData.name;
			stuFileInfo.strFileName = findData.name;
			//cout << findData.name << endl;
			append(/*listHrFileName_*/_t, stuFileInfo);
		}
			//cout << findData.name << "\t" << findData.size << " bytes.\n";
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);    // 关闭搜索句柄
}