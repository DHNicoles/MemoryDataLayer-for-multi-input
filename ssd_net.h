/*************************************************************************
  > File Name: ../include/ssd_net.h
  > Author: cheguangfu
  > Mail: cheguagnfu1@jd.com 
  > Created Time: 2017年09月11日 星期一 11时11分23秒
 ************************************************************************/

#ifndef ssd_net_h_
#define ssd_net_h_
#include <caffe/caffe.hpp>
#include "../utils/util.h"
#include "../utils/singleton.h"
using namespace caffe;
namespace ice 
{
    //Chegf: because there are two model in this project, net is unset to be a singleton!
	//class SSDNet :public Singleton<SSDNet>
	class SSDNet
	{
		public:
			SSDNet();
			~SSDNet();
			void OnInit(const std::string& model_file, const std::string& weights_file, bool is_big_model = true);
			void OnDestroy();
			void Detect(const cv::Mat& img, std::vector<cv::Rect>& head_box_v, std::vector<float>& head_score_v, float detect_thresh = 0.5);
	        void Detect(std::vector<cv::Mat>& img_v, std::vector<std::vector<cv::Rect> >& head_box_v, std::vector<std::vector<float> >& head_score_v, float detect_thresh);
		private:
			void WrapInputLayer(std::vector<cv::Mat>* input_channels);
			void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
			boost::shared_ptr<Net<float> > net_;
			cv::Size input_geometry_;
			int num_channels_;
			cv::Mat mean_;
	};

}
#endif//ssd_net_h_
