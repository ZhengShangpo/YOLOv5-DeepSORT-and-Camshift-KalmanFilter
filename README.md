# YOLOv5-DeepSORT-and-Camshift-KalmanFilter
这个项目主要涉及以下内容：
	（1）针对传统Camshfit算法在遮挡情况下无法有效跟踪目标的问题，设计了一种基于Kalman滤波的Camshift改进跟踪算法；在Kalman滤波器原状态变量中增加高维特征，解决了Kalman滤波器无法输出跟踪框的问题。实验结果表明本文算法的有效帧率和平均BH系数比传统Camshfit算法有较大提升。
	（2）针对传统机器学习目标跟踪算法精度低，鲁棒性差的问题，本文改进了YOLOv5s的损失函数和网络结构，引入了注意力机制，设计了一种基于改进YOLOv5s与DeepSORT的多目标跟踪算法。基于VOC数据集和MOT17数据集的实验结果表明本文算法比传统YOLOv5a算法在MOTA、MT、IDF1等指标均有所提升。
	（3）基于PyQt5工具包开发了目标跟踪系统软件，融合了上述两种设计的算法，实现了算法的可视化应用。
   实验结果表明本文设计的两种改进算法能有效解决现有算法中存在的一些问题，算法性能有一定提升。

<img width="361" alt="image" src="https://github.com/ZhengShangpo/YOLOv5-DeepSORT-and-Camshift-KalmanFilter/assets/91404503/e8cbbd9f-135c-4669-983e-662c5ae34e1b">
<img width="370" alt="image" src="https://github.com/ZhengShangpo/YOLOv5-DeepSORT-and-Camshift-KalmanFilter/assets/91404503/908cb446-fd88-433f-bb25-912e988673b4">
