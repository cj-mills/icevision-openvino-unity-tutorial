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
float color_list[][3] = { {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 0.5, 1.0}, {1.0, 0.5, 0.0}, {0.5, 0.75, 0.5}, {0.22306, 0.03884, 0.63863}, {0.83834, 0.02618, 0.28495}, {0.14941, 0.49409, 0.07838}, {0.87966, 0.48671, 0.89008}, {0.0, 1.0, 1.0}, {0.91938, 0.99861, 0.05139}, {0.0, 1.0, 0.5}, {0.53346, 0.91229, 0.96735}, {0.51542, 0.33546, 0.43115}, {0.57386, 0.70814, 0.00274}, {0.51983, 0.19915, 0.95881}, {0.98043, 0.82465, 0.52956}, {0.01717, 0.50508, 0.53789}, {0.44103, 0.17112, 0.0093} };
// Fridge class labels
std::string class_names[] = { "call", "no_gesture", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted" };


// The scale values used to adjust the model output to the source image resolution
float scale_x;
float scale_y;

// The minimum confidence score to consider an object proposal
float bbox_conf_thresh = 0.3;
// The maximum intersection over union value before an object proposal will be ignored
float nms_thresh = 0.45;

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
/// Generate offset values to navigate the raw model output
/// </summary>
/// <param name="height">The model input height</param>
/// <param name="width">The model input width</param>
void GenerateGridsAndStride(int height, int width)
{
	// Remove the values for the previous input resolution
	grid_strides.clear();

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
/// Generate object detection proposals from the raw model output
/// </summary>
/// <param name="out_ptr">A pointer to the output tensor data</param>
void GenerateYoloxProposals(float* out_ptr, int proposal_length)
{
	// Remove the proposals for the previous model output
	proposals.clear();

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
		float x_center = (out_ptr[start_idx + 0] + grid0) * stride;
		float y_center = (out_ptr[start_idx + 1] + grid1) * stride;

		// Get the dimensions for the predicted bounding box
		float w = exp(out_ptr[start_idx + 2]) * stride;
		float h = exp(out_ptr[start_idx + 3]) * stride;

		// Calculate the coordinates for the upper left corner of the bounding box
		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;

		// Get the confidence score that an object is present
		float box_objectness = out_ptr[start_idx + 4];

		// Initialize object struct with bounding box information
		Object obj = { x0, y0, w, h, 0, 0 };

		// Find the object class with the highest confidence score
		for (int class_idx = 0; class_idx < num_classes; class_idx++)
		{
			// Get the confidence score for the current object class
			float box_cls_score = out_ptr[start_idx + 5 + class_idx];
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
		if (obj.prob > bbox_conf_thresh) proposals.push_back(obj);
	}

	// Sort the proposals based on the confidence score in descending order
	std::sort(proposals.begin(), proposals.end(), [](Object& a, Object& b) -> bool
		{ return a.prob > b.prob; });
}

/// <summary>
/// Filter through a sorted list of object proposals using Non-maximum suppression
/// </summary>
void NmsSortedBboxes()
{
	// Remove the picked proposals for the previous model outptut
	proposal_indices.clear();

	// Iterate through the object proposals
	for (int i = 0; i < proposals.size(); i++)
	{
		Object& a = proposals[i];

		// Create OpenCV rectangle for the Object bounding box
		cv::Rect_<float> rect_a = cv::Rect_<float>(a.x0, a.y0, a.width, a.height);

		bool keep = true;

		// Check if the current object proposal overlaps any selected objects too much
		for (int j : proposal_indices)
		{
			Object& b = proposals[j];

			// Create OpenCV rectangle for the Object bounding box
			cv::Rect_<float> rect_b = cv::Rect_<float>(b.x0, b.y0, b.width, b.height);

			// Calculate the area where the two object bounding boxes overlap
			float inter_area = (rect_a & rect_b).area();
			// Calculate the union area of both bounding boxes
			float union_area = rect_a.area() + rect_b.area() - inter_area;
			// Ignore object proposals that overlap selected objects too much
			if (inter_area / union_area > nms_thresh)
				keep = false;
		}

		// Keep object proposals that do not overlap selected objects too much
		if (keep) proposal_indices.push_back(i);
	}
}

/// <summary>
/// Annotate source image with bounding boxes
/// </summary>
/// <param name="input_img"></param>
/// <param name="objects"></param>
void draw_objects(cv::Mat& image, std::vector<Object>& objects)
{
	// Make a copy of the input image
	//cv::Mat image = input_img.clone();

	// Iterate through selected proposals
	for (int i : proposal_indices)
	{
		// Get the next selected object
		Object& obj = objects[i];
		cv::Rect_<float> rect = cv::Rect_<float>(obj.x0 / scale_x, obj.y0 / scale_y, obj.width / scale_x, obj.height / scale_y);
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
		// Set label font size
		float font_size = 0.75;
		// Set label font thickness
		int font_thickness = 2;
		// Calculate the label size
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_size, font_thickness, &baseLine);
		// Clamp bounding box to image border
		int x = std::min((int)rect.x, image.cols);
		int y = std::min((int)rect.y, image.rows);
		//  Define text box
		cv::Rect text_box(cv::Point(x, y - label_size.height), cv::Size(label_size.width, label_size.height + baseLine));
		// Draw text box
		cv::rectangle(image, text_box, (color * 255), -1);
		// Add text
		cv::Mat gray_mat = cv::Mat(1, 1, CV_8UC3);
		gray_mat.setTo(color);
		cv::cvtColor(gray_mat, gray_mat, cv::COLOR_BGR2GRAY);
		cv::Scalar txt_color = gray_mat.data[0] > 0.5 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
		cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, font_size, txt_color, font_thickness);
	}

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
	cv::Mat input_image = image.clone();
	// Convert image from BGR to RGB format
	cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

	// The dimensions of the source input image
	int img_w = input_image.cols;
	int img_h = input_image.rows;
	// Calculate new input dimensions based on the max stride value
	int input_w = (int)(strides.back() * std::roundf(img_w / strides.back()));
	int input_h = (int)(strides.back() * std::roundf(img_h / strides.back()));

	// The number of color channels
	int n_channels = input_image.channels();
	// Get model input dimensions
	int n_pixels = input_w * input_h;

	// Calculate the value used to adjust the model output to the source image resolution
	scale_x = input_w / (img_w * 1.0);
	scale_y = input_h / (img_h * 1.0);
	// Resize input image to new dimensions
	cv::resize(input_image, input_image, cv::Size(input_w, input_h));

	// Generate the grid and stride values based on input resolution
	GenerateGridsAndStride(input_h, input_w);

	// Initialize an OpenVINO runtime core entity
	ov::Core core;
	// Enable caching for compiled models
	core.set_property("GPU", ov::cache_dir("cache"));
	// Get the names of available compute devices
	std::cout << "\nAvailable Devices: ";
	std::vector<std::string> available_devices = core.get_available_devices();
	for (std::string device : available_devices) std::cout << device << " ";
	std::cout << "\n\n";

	// Read in a model file
	std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
	// Update model input dimensions
	model->reshape({ 1, 3, input_image.rows, input_image.cols });
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
	
	// Start timer for measuring inference speed
	auto start = std::chrono::high_resolution_clock::now();

	// Iterate over each pixel in image
	for (int p = 0; p < n_pixels; p++)
	{
		input_data[0 * n_pixels + p] = input_image.data[p * n_channels + 0] / 255.0f;
		input_data[1 * n_pixels + p] = input_image.data[p * n_channels + 1] / 255.0f;
		input_data[2 * n_pixels + p] = input_image.data[p * n_channels + 2] / 255.0f;
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

	// Generate new proposals for the current model output
	GenerateYoloxProposals(out_data, output_tensor.get_shape()[2]);

	// Pick detected objects to keep using Non-maximum Suppression
	NmsSortedBboxes();
	// Get the number of objects detected
	printf("Detected %d objects\n", (int)proposal_indices.size());
	
	// Annotate source image with model predictions
	draw_objects(image, proposals);
}
