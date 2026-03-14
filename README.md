# HITIRC-Spring2026
This is a work-submitting repository for tasks of HITIRC in spring,2026.All these results thank for the owners,Uin and Lxl,also with aids from AI assistants like Qwen .

由于设备原因kuavo任务二尚未完成，我们目前已经完成了除kuavo任务二（MPC）之外的所有任务。

针对Aelos任务，代码和识别结果均保存在code中的task1文件夹中；Roban任务相对于Aelos多了SLAM部分，我们已经初步学习了SLAM的基本框架，目前正在进行ORB-SLAM2的一些实战。

对于Kuavo任务，我们已经完成了任务二基础部分，实现了通过自主编写的脚本向/cmd_vel话题发送控制信息，具体的代码保存在move.py，可以实现通过aswd控制机器人移动方向，演示视频为。

针对任务三，我们完成了yolo模型的训练，不仅实现了识别数据集中的门把手，还成功识别了寝室的门把手，其中，对于测试集的识别效果以.jpg格式保存在code中的val_batch0_pred.jpg,val_batch1_pred.jpg,val_batch2_pred.jpg，源代码保存在train.py中，模型为best.pt,加入卡尔曼滤波的版本保存在track_with_kalman.py中，效果视频保存在1f221627607e6617f86b1894c2e5552e.mp4。


我们使用的配置为：Ultralytics: 8.4.21，PyTorch: 2.4.1+cu121，CUDA: 12.1，OpenCV: 4.13.0，YOLOv8.

错误整理：任务二进行初期，我们遇到了诸多问题，如docker连接不上，仓库下载错误等问题，这启发我们将眼前的错误归结为一个具体的原因，针对具体的原因寻求解决办法，这种归类的方式解决问题十分高效；同时我们意识到在进行实际行动之前要对任务的材料进行充分的了解，不然会南辕北辙，事倍功半；在解决MPC的过程中，我们遇到了代码位置错误，docker使用错误等诸多细枝末节的阻挠，使我们深刻到清晰的项目结构对于项目正常进行的重要作用。

