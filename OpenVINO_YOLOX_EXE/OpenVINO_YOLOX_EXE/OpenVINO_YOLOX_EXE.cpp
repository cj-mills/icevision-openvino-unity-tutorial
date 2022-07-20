// OpenVINO_2022_YOLOX_EXE.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

// Stores information about a single object prediction
struct Object
{
	float x0;
	float y0;
	float width;
	float height;
	int label;
	float prob;
};

// Store grid offset and stride values to decode a section of the model output
struct GridAndStride
{
	int grid0;
	int grid1;
	int stride;
};

// RGB color list
//float color_list[][3] = { {0.88319, 0.08311, 0.71191}, {0.63624, 0.10606, 0.55944}, {0.49306, 0.21503, 0.05323}, {0.28165, 0.48839, 0.91077}, {0.2147, 0.48966, 0.19114}, {0.30997, 0.83882, 0.932}, {0.21612, 0.77331, 0.73153}, {0.79444, 0.74729, 0.46322}, {0.66057, 0.99022, 0.60685}, {0.51231, 0.98009, 0.59455}, {0.50289, 0.08377, 0.47362}, {0.15457, 0.48123, 0.24212}, {0.83821, 0.3014, 0.93705}, {0.98649, 0.73279, 0.07521}, {0.80319, 0.42785, 0.22765}, {0.3892, 0.47654, 0.69036}, {0.76941, 0.70456, 0.36975}, {0.90023, 0.078, 0.13737}, {0.42451, 0.8784, 0.5826}, {0.73633, 0.54653, 0.18163}, {0.34936, 0.85942, 0.77696}, {0.38348, 0.46305, 0.44919}, {0.01876, 0.43492, 0.08287}, {0.47295, 0.39229, 0.95749}, {0.42301, 0.77877, 0.71532}, {0.60035, 0.00444, 0.1745}, {0.04167, 0.21011, 0.06148}, {0.21677, 0.12897, 0.71129}, {0.89641, 0.06952, 0.93395}, {0.17959, 0.14552, 0.97496}, {0.47082, 0.94568, 0.75006}, {0.6623, 0.0842, 0.42559}, {0.74814, 0.9088, 0.95614}, {0.64544, 0.37419, 0.33494}, {0.61622, 0.8261, 0.22977}, {0.90729, 0.50626, 0.2295}, {0.44649, 0.06256, 0.44934}, {0.48083, 0.29358, 0.03613}, {0.56928, 0.46334, 0.97285}, {0.86787, 0.12326, 0.68888}, {0.42263, 0.18191, 0.15364}, {0.99155, 0.81773, 0.79963}, {0.90042, 0.27848, 0.71969}, {0.40038, 0.21564, 0.55661}, {0.53565, 0.04282, 0.20631}, {0.45282, 0.68609, 0.67031}, {0.811, 0.55726, 0.28195}, {0.79888, 0.03662, 0.43465}, {0.55336, 0.03145, 0.69141}, {0.04626, 0.89695, 0.3341}, {0.7764, 0.42498, 0.28402}, {0.89231, 0.79846, 0.84397}, {0.60304, 0.62812, 0.70791}, {0.57327, 0.51147, 0.63946}, {0.89851, 0.96925, 0.56361}, {0.39946, 0.44608, 0.19465}, {0.55684, 0.24583, 0.4151}, {0.89894, 0.34337, 0.31515}, {0.20124, 0.20545, 0.54863}, {0.77877, 0.0553, 0.82661}, {0.03154, 0.07404, 0.01619}, {0.55641, 0.02001, 0.7698}, {0.9016, 0.20665, 0.16741}, {0.06733, 0.58359, 0.64599}, {0.07426, 0.73879, 0.48413}, {0.95478, 0.73805, 0.73276}, {0.29825, 0.73915, 0.44375}, {0.05566, 0.34968, 0.10582}, {0.96384, 0.85364, 0.47936}, {0.3869, 0.42031, 0.28212}, {0.08473, 0.47985, 0.32686}, {0.63524, 0.752, 0.70884}, {0.27848, 0.66611, 0.53699}, {0.96936, 0.40184, 0.77043}, {0.23958, 0.80604, 0.90901}, {0.34097, 0.54933, 0.43285}, {0.9198, 0.09017, 0.34672}, {0.12151, 0.17597, 0.2081}, {0.88528, 0.87222, 0.87124}, {0.24166, 0.69045, 0.10446} };
// COCO class labels
//std::string class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

// RGB color list
float color_list[][3] = { {0.89009, 0.03805, 0.30262}, {0.99718, 0.65819, 0.51767}, {0.58328, 0.34627, 0.61686}, {0.98407, 0.67736, 0.54331} };
// Fridge class labels
std::string class_names[] = { "carton", "milk_bottle", "can", "water_bottle" };


// The scale value used to adjust the model output to the original unpadded image
float scaleX;
float scaleY;

// The minimum confidence score to consider an object proposal
float bbox_conf_thresh = 0.3;
// The maximum intersection over union value before an object proposal will be ignored
float nms_thresh = 0.45;

// The mean of the ImageNet dataset used to train the model
float mean[] = { 0.485, 0.456, 0.406 };
// The standard deviation of the ImageNet dataset used to train the model
float std_dev[] = { 0.229, 0.224, 0.225 };

// List of available compute devices
std::vector<std::string> available_devices;
// Stores the grid and stride values
std::vector<GridAndStride> grid_strides;
// Stores the object proposals with confidence scores above bbox_conf_thresh
std::vector<Object> proposals;
// Stores the indices for the object proposals selected using non-maximum suppression
std::vector<int> proposal_indices;

// The stride values used to generate the gride_strides vector
std::vector<int> strides = { 8, 16, 32 };

/// <summary>
/// Print information about model input and output
/// </summary>
/// <param name="network"></param>
void printInputAndOutputsInfo(ov::Model& network)
{
	// Try to get a friendly name for the model
	std::cout << std::format("model name: {}\n", network.get_friendly_name());
	// Iterate through model inputs
	for (ov::Output<ov::Node> input : network.inputs())
	{
		printf("%4sinput\n", "");
		printf("%8sinput name: %s\n", "", input.get_names().empty() ? "NONE" : input.get_any_name().c_str());
		std::cout << std::format("{:8s}input type: ", "") << input.get_element_type() << std::endl;
		std::cout << std::format("{:8s}input shape: ", "") << input.get_shape() << std::endl;
	}
	// Iterate through model outputs
	for (ov::Output<ov::Node> output : network.outputs())
	{
		printf("%4soutput\n", "");
		printf("%8soutput name: %s\n", "", output.get_names().empty() ? "NONE" : output.get_any_name().c_str());
		std::cout << std::format("{:8s}output type: ", "") << output.get_element_type() << std::endl;
		std::cout << std::format("{:8s}output shape: ", "") << output.get_shape() << std::endl;
	}
	std::cout << std::endl;
}

/// <summary>
/// Generate grid and stride values
/// </summary>
/// <param name="height"></param>
/// <param name="width"></param>
void GenerateGridsAndStride(int height, int width)
{
	// Iterate through each stride value
	for (auto stride : strides)
	{
		// Calculate the grid dimensions
		int grid_height = height / stride;
		int grid_width = width / stride;

		// Store each combination of grid coordinates
		for (int g1 = 0; g1 < grid_height; g1++)
		{
			for (int g0 = 0; g0 < grid_width; g0++)
			{
				grid_strides.push_back(GridAndStride{ g0, g1, stride });
			}
		}
	}
}

/// <summary>
/// Create object proposals for all model predictions with high enough confidence scores
/// </summary>
/// <param name="feat_ptr"></param>
void GenerateYoloxProposals(float* feat_ptr, int proposal_length)
{
	// Obtain the number of classes the model was trained to detect
	int num_classes = proposal_length - 5;

	for (int anchor_idx = 0; anchor_idx < grid_strides.size(); anchor_idx++)
	{
		// Get the current grid and stride values
		int grid0 = grid_strides[anchor_idx].grid0;
		int grid1 = grid_strides[anchor_idx].grid1;
		int stride = grid_strides[anchor_idx].stride;

		// Get the starting index for the current proposal
		int start_idx = anchor_idx * proposal_length;

		// Get the coordinates for the center of the predicted bounding box
		float x_center = (feat_ptr[start_idx + 0] + grid0) * stride;
		float y_center = (feat_ptr[start_idx + 1] + grid1) * stride;

		// Get the dimensions for the predicted bounding box
		float w = exp(feat_ptr[start_idx + 2]) * stride;
		float h = exp(feat_ptr[start_idx + 3]) * stride;

		// Calculate the coordinates for the upper left corner of the bounding box
		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;

		// Get the confidence score that an object is present
		float box_objectness = feat_ptr[start_idx + 4];

		// Initialize object struct with bounding box information
		Object obj = { x0, y0, w, h, 0, 0 };

		// Find the object class with the highest confidence score
		for (int class_idx = 0; class_idx < num_classes; class_idx++)
		{
			// Get the confidence score for the current object class
			float box_cls_score = feat_ptr[start_idx + 5 + class_idx];
			// Calculate the final confidence score for the object proposal
			float box_prob = box_objectness * box_cls_score;

			// Check for the highest confidence score
			if (box_prob > obj.prob)
			{
				obj.label = class_idx;
				obj.prob = box_prob;
			}
		}

		// Only add object proposals with high enough confidence scores
		if (obj.prob > bbox_conf_thresh)
			proposals.push_back(obj);
	}
}

/// <summary>
/// Filter through a sorted list of object proposals using Non-maximum suppression
/// </summary>
void NmsSortedBboxes()
{
	// Iterate through the object proposals
	for (int i = 0; i < proposals.size(); i++)
	{
		Object& a = proposals[i];

		// Create OpenCV rectangle for the Object bounding box
		cv::Rect_<float> aRect = cv::Rect_<float>(a.x0, a.y0, a.width, a.height);
		// Get the bounding box area
		float aRect_area = aRect.area();

		bool keep = true;

		// Check if the current object proposal overlaps any selected objects too much
		for (int j : proposal_indices)
		{
			Object& b = proposals[j];

			// Create OpenCV rectangle for the Object bounding box
			cv::Rect_<float> bRect = cv::Rect_<float>(b.x0, b.y0, b.width, b.height);

			// Calculate the area where the two object bounding boxes overlap
			float inter_area = (aRect & bRect).area();
			// Calculate the union area of both bounding boxes
			float union_area = aRect_area + bRect.area() - inter_area;
			// Ignore object proposals that overlap selected objects too much
			if (inter_area / union_area > nms_thresh)
				keep = false;
		}

		// Keep object proposals that do not overlap selected objects too much
		if (keep)
			proposal_indices.push_back(i);
	}
}

/// <summary>
/// Annotate source image with bounding boxes
/// </summary>
/// <param name="input_img"></param>
/// <param name="objects"></param>
void draw_objects(cv::Mat& input_img, std::vector<Object>& objects)
{
	// Make a copy of the input image
	cv::Mat image = input_img.clone();
	// Iterate through selected proposals
	for (int i : proposal_indices)
	{
		// Get the next selected object
		Object& obj = objects[i];
		cv::Rect_<float> rect = cv::Rect_<float>(obj.x0 / scaleX, obj.y0 / scaleY, obj.width / scaleX, obj.height / scaleY);
		// Get the predicted class label and associated confidence value
		std::string label = class_names[obj.label];
		float confidence = obj.prob * 100.0;
		// Print prediction information
		std::string output_str = "Label: %s, Conf: %.2f%%, Coords: (%.2f,%.2f), Box: (%.2fx%.2f)\n";
		printf(output_str.c_str(), label.c_str(), confidence, rect.x, rect.y, rect.width, rect.height);
		// Get the associated bounding box color for the predicted object class
		cv::Scalar color(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
		// Draw bounding box
		cv::rectangle(image, rect, color * 255, 2);
		// Define bounding box text
		std::string text = std::format("{} {:.1f}%", label, confidence);
		// Get text box size
		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
		// Clamp bounding box to image border
		int x = std::min((int)rect.x, image.cols);
		int y = std::min((int)rect.y, image.rows);
		//  Define text box
		cv::Rect text_box(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine));
		// Draw text box
		cv::rectangle(image, text_box, (color * 0.7 * 255), -1);
		// Add text
		cv::Scalar txt_color = cv::mean(color)[0] > 0.5 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
		cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
	}

	// Convert input image back to BGR format
	cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	// Save annotated image
	std::string out_name = "output.jpg";
	cv::imwrite(out_name, image);
	std::cout << std::format("\nSaved annotated image to {}\n", out_name);
}

/// <summary>
/// Use a YOLOX model to detect objects in a test image using the specified compute device
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>
int main(int argc, const char* argv[])
{
	// Load an input image
	cv::Mat image = cv::imread(argv[2]);
	// Convert image from BGR to RGB format
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// The dimensions of the source input image
	int img_w = image.cols;
	int img_h = image.rows;
	// Calculate new input dimensions based on the max stride value
	int input_w = (int)(strides.back() * std::roundf(img_w / strides.back()));
	int input_h = (int)(strides.back() * std::roundf(img_h / strides.back()));
	// Calculate the value used to adjust the model output to the source image resolution
	scaleX = input_w / (img_w * 1.0);
	scaleY = input_h / (img_h * 1.0);
	// Resize input image to new dimensions
	cv::resize(image, image, cv::Size(input_w, input_h));

	// Generate the grid and stride values based on input resolution
	GenerateGridsAndStride(input_h, input_w);

	// Initialize an OpenVINO runtime core entity
	ov::Core core;
	// Enable caching for compiled models
	core.set_property("GPU", ov::cache_dir("cache"));
	// Get the names of available compute devices
	std::cout << "\nAvailable Devices: ";
	std::vector<std::string> available_devices = core.get_available_devices();
	for (std::string device : available_devices)
		std::cout << device << " ";
	std::cout << "\n\n";

	// Read in a model file
	std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
	// Update model input dimensions
	model->reshape({ 1, 3, image.rows, image.cols });
	// Print model input and output
	printInputAndOutputsInfo(*model);
	// Compile and load network to device
	auto compiled_model = core.compile_model(model, "MULTI",
		ov::device::priorities(argv[3]),
		ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
		ov::hint::inference_precision(ov::element::f32));

	// Create an inference request
	ov::InferRequest infer_request = compiled_model.create_infer_request();
	// Get input tensor by index
	ov::Tensor input_tensor = infer_request.get_input_tensor(0);
	// Get a pointer to the input tensor data
	float* input_data = input_tensor.data<float>();

	// The number of color channels
	int num_channels = image.channels();
	// Get model input dimensions
	int H = input_tensor.get_shape()[2];
	int W = input_tensor.get_shape()[3];
	int nPixels = W * H;

	// Start timer for measuring inference speed
	auto start = std::chrono::high_resolution_clock::now();

	// Iterate over each pixel in image
	for (int p = 0; p < nPixels; p++)
	{
		// Iterate over each color channel for each pixel in image
		for (int ch = 0; ch < num_channels; ++ch)
		{
			int source_idx = p * num_channels + ch;
			int dest_idx = ch * nPixels + p;
			input_data[dest_idx] = (image.data[source_idx] / 255.0f - mean[ch]) / std_dev[ch];
		}
	}

	// Perform inference
	infer_request.infer();

	// Get model output tensor
	ov::Tensor output_tensor = infer_request.get_output_tensor();
	// Get a pointer to the output tensor data
	float* out_data = output_tensor.data<float>();

	// Stop timer for measuring inference speed
	auto stop = std::chrono::high_resolution_clock::now();
	// Calculate inference speed
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	float fps = (1000.0 / duration.count());
	printf("Inference time: %dms (i.e., %.2ffps)\n\n", (int)duration.count(), fps);

	// Remove the proposals for the previous model output
	proposals.clear();
	// Generate new proposals for the current model output
	GenerateYoloxProposals(out_data, output_tensor.get_shape()[2]);

	// Sort the generated proposals based on their confidence scores
	auto compare_func = [](Object& a, Object& b) -> bool
	{ return a.prob > b.prob; };
	std::sort(proposals.begin(), proposals.end(), compare_func);

	// Remove the picked proposals for the previous model outptut
	proposal_indices.clear();
	// Pick detected objects to keep using Non-maximum Suppression
	NmsSortedBboxes();
	// Get the number of objects detected
	printf("Detected %d objects\n", (int)proposal_indices.size());
	// Resize input image back to source resolution
	cv::resize(image, image, cv::Size(img_w, img_h));
	// Annotate input image with model predictions
	draw_objects(image, proposals);
}
