// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

	// Inference engine instance
	ov::Core core;
	// The user define model representation
	std::shared_ptr<ov::Model> model;
	// A device-specific compiled model
	ov::CompiledModel compiled_model;

	// List of available compute devices
	std::vector<std::string> available_devices;
	// An inference request for a compiled model
	ov::InferRequest infer_request;
	// Stores the model input data
	ov::Tensor input_tensor;
	// A pointer for accessing the input tensor data
	float* input_data;

	// model has only one output
	ov::Tensor output_tensor;
	// A pointer for accessing the output tensor data
	float* out_data;

	// The current source image width
	int img_w;
	// The current source image height
	int img_h;
	// The current model input width
	int input_w;
	// The current model input height
	int input_h;
	// The total number pixels in the input image
	int n_pixels;
	// The number of color channels 
	int num_channels = 3;

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

	// The scale value used to adjust the model output to the original unpadded image
	float scale_x;
	float scale_y;

	// The minimum confidence score to consider an object proposal
	float bbox_conf_thresh = 0.3;
	// The maximum intersection over union value before an object proposal will be ignored
	float nms_thresh = 0.45;

	// The mean of the ImageNet dataset used to train the model
	float mean[] = { 0.485, 0.456, 0.406 };
	// The standard deviation of the ImageNet dataset used to train the model
	float std_dev[] = { 0.229, 0.224, 0.225 };

	// Stores the grid and stride values
	std::vector<GridAndStride> grid_strides;
	// Stores the object proposals with confidence scores above bbox_conf_thresh
	std::vector<Object> proposals;
	// Stores the indices for the object proposals selected using non-maximum suppression
	std::vector<int> proposal_indices;

	// The stride values used to generate the gride_strides vector
	std::vector<int> strides = { 8, 16, 32 };

	/// <summary>
	/// Get the number of available compute devices
	/// </summary>
	/// <returns>The number of available devices</returns>
	DLLExport int GetDeviceCount() 
	{
		// Reset list of available compute devices
		available_devices.clear();

		// Populate list of available compute devices
		for (std::string device : core.get_available_devices()) {
			// Skip GNA device
			if (device.find("GNA") == std::string::npos) {
				available_devices.push_back(device);
			}
		}
		// Return the number of available compute devices
		return available_devices.size();
	}

	/// <summary>
	/// Get the name of the compute device name at the specified index
	/// </summary>
	/// <param name="index"></param>
	/// <returns>The name of the device at the specified index</returns>
	DLLExport std::string* GetDeviceName(int index) {
		return &available_devices[index];
	}


	/// <summary>
	/// Generate grid and stride values
	/// </summary>
	/// <param name="height">The model input height</param>
	/// <param name="width">The model input width</param>
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
	/// Set minimum confidence score for keeping bounding box proposals
	/// </summary>
	/// <param name="min_confidence">The minimum confidence score for keeping bounding box proposals</param>
	DLLExport void SetConfidenceThreshold(float min_confidence)
	{
		bbox_conf_thresh = min_confidence;
	}


	/// <summary>
	/// Load a model from the specified file path
	/// </summary>
	/// <param name="model_path">The full model path to the OpenVINO IR model</param>
	/// <param name="index">The index for the available_devices vector</param>
	/// <param name="image_dims">The source image dimensions</param>
	/// <returns>A status value indicating success or failure to load and reshape the model</returns>
	DLLExport int LoadModel(char* model_path, int index, int image_dims[2]) 
	{

		int return_val = 0;
		// Set the cache directory for compiled GPU models
		core.set_property("GPU", ov::cache_dir("cache"));

		// Try loading the specified model
		try { model = core.read_model(model_path); }
		catch (...) { return 1; }

		// The dimensions of the source input image
		img_w = image_dims[0];
		img_h = image_dims[1];
		// Calculate new input dimensions based on the max stride value
		input_w = (int)(strides.back() * std::roundf(img_w / strides.back()));
		input_h = (int)(strides.back() * std::roundf(img_h / strides.back()));
		n_pixels = input_w * input_h;

		// Calculate the value used to adjust the model output to the source image resolution
		scale_x = input_w / (img_w * 1.0);
		scale_y = input_h / (img_h * 1.0);

		// Generate the grid and stride values based on input resolution
		grid_strides.clear();
		GenerateGridsAndStride(input_h, input_w);

		// Try updating the model input dimensions
		try { model->reshape({ 1, 3, input_h, input_w }); }
		catch (...) { return_val = 2; }

		// Create a compiled model for the taret compute device
		auto compiled_model = core.compile_model(model, "MULTI",
			ov::device::priorities(available_devices[index]),
			ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
			ov::hint::inference_precision(ov::element::f32));

		// Create an inference request
		infer_request = compiled_model.create_infer_request();

		// Get input tensor by index
		input_tensor = infer_request.get_input_tensor(0);
		// Get a pointer to the input tensor data
		input_data = input_tensor.data<float>();
		 
		// Get output tensor
		output_tensor = infer_request.get_output_tensor();
		// Get a pointer to the output tensor data
		out_data = output_tensor.data<float>();

		// Replace the initial input dims with the updated values
		image_dims[0] = input_w;
		image_dims[1] = input_h;

		// Return a value of 0 if the model loads successfully
		return return_val;
	}

	/// <summary>
	/// Create object proposals for all model predictions with high enough confidence scores
	/// </summary>
	/// <param name="feat_ptr">A pointer to the output tensor data</param>
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
			if (keep)
				proposal_indices.push_back(i);
		}
	}


	/// <summary>
	/// Perform inference with the provided texture data
	/// </summary>
	/// <param name="image_data">The source image data from Unity</param>
	/// <returns>The final number of detected objects</returns>
	DLLExport int PerformInference(uchar* image_data) 
	{

		// Store the pixel data for the source input image in an OpenCV Mat
		cv::Mat input_image = cv::Mat(img_h, img_w, CV_8UC4, image_data);

		// Remove the alpha channel
		cv::cvtColor(input_image, input_image, cv::COLOR_RGBA2RGB);

		// Resize the input image
		cv::resize(input_image, input_image, cv::Size(input_w, input_h));

		// Iterate over each pixel in image
		for (int p = 0; p < n_pixels; p++)
		{
			// Iterate over each color channel for each pixel in image
			for (int ch = 0; ch < num_channels; ++ch)
			{
				int source_idx = p * num_channels + ch;
				int dest_idx = ch * n_pixels + p;
				input_data[dest_idx] = (input_image.data[source_idx] / 255.0f);
			}
		}

		// Perform inference
		infer_request.infer();

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

		// return the final number of detected objects
		return (int)proposal_indices.size();
	}

	/// <summary>
	/// Fill the provided array with the detected objects
	/// </summary>
	/// <param name="objects">A pointer to a list of objects from Unity</param>
	DLLExport void PopulateObjectsArray(Object* objects) 
	{

		for (int i = 0; i < proposal_indices.size(); i++)
		{
			Object obj = proposals[proposal_indices[i]];

			// Adjust offset to source image resolution and clamp the bounding box
			objects[i].x0 = std::min(obj.x0 / scale_x, (float)img_w);
			objects[i].y0 = std::min(obj.y0 / scale_y, (float)img_h);
			objects[i].width = std::min(obj.width / scale_x, (float)img_w);
			objects[i].height = std::min(obj.height / scale_y, (float)img_h);

			objects[i].label = obj.label;
			objects[i].prob = obj.prob;
		}
	}

	/// <summary>
	/// Reset vectors
	/// </summary>
	DLLExport void FreeResources() 
	{
		available_devices.clear();
		grid_strides.clear();
		proposals.clear();
		proposal_indices.clear();
	}
}
