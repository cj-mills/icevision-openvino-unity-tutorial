# IceVision → OpenVINO → Unity Tutorial

## Tutorial Links
* [Part 1](https://christianjmills.com/posts/icevision-openvino-unity-tutorial/part-1/): Train a YOLOX model using IceVision and export it to OpenVINO. 
* [Part 2](https://christianjmills.com/posts/icevision-openvino-unity-tutorial/part-2/): Create a dynamic link library (DLL) file in Visual Studio to perform object detection with a YOLOX model using OpenVINO. 
* [Part 3](https://christianjmills.com/posts/icevision-openvino-unity-tutorial/part-3/):  Perform object detection in a Unity project with OpenVINO. 
* [Follow up](https://christianjmills.com/posts/onnx-directml-unity-tutorial/part-1/): Use ONNX Runtime and DirectML instead of OpenVINO.

## Demo Video
https://user-images.githubusercontent.com/9126128/183220227-868d552b-c67e-48b6-97f9-433c5634230a.mp4

## Training Code

| Jupyter Notebook                                             | Colab                                                        | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kaggle&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/icevision-openvino-unity-tutorial/blob/main/notebooks/Icevision-YOLOX-to-OpenVINO-Tutorial-HaGRID.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cj-mills/icevision-openvino-unity-tutorial/blob/main/notebooks/Icevision-YOLOX-to-OpenVINO-Tutorial-HaGRID-Colab.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/cj-mills/icevision-openvino-unity-tutorial/blob/main/notebooks/Icevision-YOLOX-to-OpenVINO-Tutorial-HaGRID-Kaggle.ipynb) |

**Note:** Training on the free GPU tier for Google Colab takes approximately 11 minutes per epoch, while training on the free GPU tier for Kaggle Notebooks takes around 15 minutes per epoch.



## Kaggle Datasets

* [HaGRID Sample 30k 384p](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p)
* [HaGRID Sample 120k 384p](https://www.kaggle.com/datasets/innominate817/hagrid-sample-120k-384p)


<details><summary><h2>Reference Images</h2></summary><br/>

| Class    | Image                                              |
| --------- | ------------------------------------------------------------ |
| call    | ![call](./images/call.jpg) |
| dislike         | ![dislike](./images/dislike.jpg) |
| fist    | ![ fist](./images/fist.jpg) |
| four         | ![four](./images/four.jpg) |
| like         | ![ like](./images/like.jpg) |
| mute         | ![ mute](./images/mute.jpg) |
| ok    | ![ ok](./images/ok.jpg) |
| one         | ![ one](./images/one.jpg) |
| palm         | ![ palm](./images/palm.jpg) |
| peace         | ![peace](./images/peace.jpg) |
| peace_inverted         | ![peace_inverted](./images/peace_inverted.jpg) |
| rock         | ![rock](./images/rock.jpg) |
| stop         | ![stop](./images/stop.jpg) |
| stop_inverted         | ![stop_inverted](./images/stop_inverted.jpg) |
| three         | ![three](./images/three.jpg) |
| three2         | ![three2](./images/three2.jpg) |
| two_up         | ![ two_up](./images/two_up.jpg) |
| two_up_inverted         | ![two_up_inverted](./images/two_up_inverted.jpg) |
</details>
