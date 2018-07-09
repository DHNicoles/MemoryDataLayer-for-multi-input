/*************************************************************************
  > File Name: ../src/ssd_net.cpp
  > Author: cheguangfu
  > Mail: cheguangfu1@jd.com 
  > Created Time: 2017年09月11日 星期一 11时33分10秒
 ************************************************************************/
#include "caffe/layers/memory_data_layer.hpp"
#include "detector/ssd_net.h"

namespace ice
{
	SSDNet::SSDNet()
	{

	}

	SSDNet::~SSDNet()
	{
		OnDestroy();
	}
	void SSDNet::OnInit(const std::string& model_file, const std::string& weights_file, bool is_big_model)
	{
#ifndef CPU_ONLY
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(0);
		/* Load the network. */
		net_.reset(new Net<float>(model_file, TEST, Caffe::GetDefaultDevice()));
#else
		Caffe::set_mode(Caffe::CPU);
		net_.reset(new Net<float>(model_file, TEST));

#endif
		net_->CopyTrainedLayersFrom(weights_file);

		//CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
		//CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

        if(is_big_model)
        {

            Blob<float>* input_layer = net_->input_blobs()[0];
            num_channels_ = input_layer->channels();
            //CHECK(num_channels_ == 3 || num_channels_ == 1)
            //	<< "Input layer should have 1 or 3 channels.";
            input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        }
	}
	void SSDNet::OnDestroy()
	{

	}

	void SSDNet::Detect(std::vector<cv::Mat>& img_v, std::vector<std::vector<cv::Rect> >& head_box_v, std::vector<std::vector<float> >& head_score_v, float detect_thresh)
    {
		static boost::shared_ptr<MemoryDataLayer<float> >  mem_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net_->layers()[0]);
		static int input_width = mem_data_layer->width();
		static int input_height = mem_data_layer->height();
        static Stopwatch T("forward");
        std::vector<int> int_vec;
        std::vector<cv::Mat> sample_v(img_v.size());
        for(int i=0;i<img_v.size();++i)
        {
            cv::resize(img_v[i], sample_v[i], cv::Size(input_width, input_height)); 
            int_vec.push_back(i);
        }

        LOG(INFO) << "Batch size: " << img_v.size();
		mem_data_layer->set_batch_size(img_v.size());
        mem_data_layer->AddMatVector(sample_v, int_vec);
    
        T.Reset();   T.Start();
		net_->Forward();
        T.Stop();
        LOG(INFO) << "local model forward cost: " << T.GetTime();

        head_box_v.resize(img_v.size());
        head_score_v.resize(img_v.size());
        /* Copy the output layer to a std::vector */
		Blob<float>* result_blob = net_->output_blobs()[0];
		const float* result = result_blob->cpu_data();
		const int num_det = result_blob->height();
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		for (int k = 0; k < num_det; ++k) 
		{
            int label = result[0];

			if (label == -1 || label >= img_v.size()) {
				// Skip invalid detection.
				result += 7;
				continue;
			}
			if(result[1] == 1 && result[2] > detect_thresh)//head and thresh
			{
                LOG(INFO) <<"[xmin,ymin,xmax,ymax]: [" << result[3] << ","  << result[4] << "," << result[5] << "," << result[6] <<"]";
				cv::Rect box =  cv::Rect(cv::Point(result[3] * img_v[label].cols, result[4] * img_v[label].rows),cv::Point(result[5] * img_v[label].cols, result[6] * img_v[label].rows));
				head_box_v[label].push_back(box);
				head_score_v[label].push_back(result[2]);
			}
			result += 7;
		}
    }
	void SSDNet::Detect(const cv::Mat& img, std::vector<cv::Rect>& head_box_v, std::vector<float>& head_score_v, float detect_thresh)
	{
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
				input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		Preprocess(img, &input_channels);

		net_->Forward();

		/* Copy the output layer to a std::vector */
		Blob<float>* result_blob = net_->output_blobs()[0];
		const float* result = result_blob->cpu_data();
		const int num_det = result_blob->height();
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		for (int k = 0; k < num_det; ++k) 
		{
			if (result[0] == -1) {
				// Skip invalid detection.
				result += 7;
				continue;
			}
			if(result[1] == 1 && result[2] > detect_thresh)//head and thresh
			{
				cv::Rect box =  cv::Rect(cv::Point(result[3] * img.cols, result[4] * img.rows),cv::Point(result[5] * img.cols, result[6] * img.rows));
				head_box_v.push_back(box);
				head_score_v.push_back(result[2]);
			}
			result += 7;
		}
	}
	/* Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel). This way we save one memcpy operation and we
	 * don't need to rely on cudaMemcpy2D. The last preprocessing
	 * operation will write the separate channels directly to the input
	 * layer. */
	void SSDNet::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
	{
		Blob<float>* input_layer = net_->input_blobs()[0];

		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}

	void SSDNet::Preprocess(const cv::Mat& img,
			std::vector<cv::Mat>* input_channels) {
		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else
			sample_resized.convertTo(sample_float, CV_32FC1);

		cv::Scalar mean(127.5, 127.5, 127.5);
		sample_float -= mean;
		cv::Mat sample_normalized;
		cv::divide(sample_float, 127.0, sample_normalized);

		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the cv::Mat
		 * objects in input_channels. */
		cv::split(sample_normalized, *input_channels);

		CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
				== net_->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
	}


}
