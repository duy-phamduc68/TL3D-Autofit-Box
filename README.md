## CCTV Inverse Perspective Mapping Tool

![app](/media/app.png)

This is the calibration tool from my bigger project [TrafficLab 3D](https://github.com/duy-phamduc68/TrafficLab-3D). Its main use is Inverse Perspective Mapping (IPM) for a traffic CCTV / Webcam and its satellite image to establish a 2-way projection between them.

The produced results include calibration data (distortion, homography...), camera pose, scale, and more, are stored as "G-projection" `.json` files.

For more context, check out my broader [blog post on TrafficLab 3D](https://yuk068.github.io/2026/02/20/traffilclab-3d-overview) and find the sections about "G-Projection".

I also have a [Guide Youtube Video for TrafficLab 3D](https://www.youtube.com/watch?v=PeL2v1YEdYA) that do covers how to use this tool.

For now, its mostly manual work, I extracted this module into a separate repo to hopefully be able to improve this process with more robust math and automated methods.

If you have any suggestions / proposed improvements, open an issue/pull request!