## Introduction:

This code repository for the paper:  
**I^2R-Net: Intra- and Inter-Human Relation Network for Multi-Person Pose Estimation**  
[Yiwei Ding, Wenjin Deng, Yinglin Zheng, Pengfei Liu, Meihong Wang, Xuan Cheng, Jianmin Bao, Dong Chen, Ming Zeng]  

![teaser](figs/pipeline.png)

![teaser](figs/vis_attention.png)

## Model Zoo

### Results on CrowdPose testing set

|     Model      | Input size |  AP    | Ap .5 | AP .75 |  AR    | AR .5  | AR .75 | AP easy | AP medium | AP hard | Download | Log |
| :------------: | :--------: |  ----- | ----- | :----: | :----: | :----: | :----: | :-----: | :-------: | :-----: | :------: | --- |
| I2R-Net (Vanilla version, 1st stage:HRNet-W48-S) |  256x192  | 0.723 | 0.924 | 0.779  | 0.765  | 0.932 | 0.819 | 0.799 | 0.732  | 0.628 | [model](#) | [log](#) |
| I2R-Net (1st stage:TransPose-H) |  256x192  | 0.763 | 0.935 | 0.822  | 0.791  | 0.940 | 0.844 | 0.832 | 0.770  | 0.674 | [model](#) | [log](#) |
| I2R-Net (1st stage:HRFormer-B) |  256x192  | 0.774 | 0.936 | 0.833  | 0.803  | 0.945 | 0.855 | 0.838 | 0.781  | 0.693 | [model](#) | [log](#) |


### Results on OCHuman valiadation set

|     Model      | Input size |  AP    | Ap .5 | AP .75 | Download | Log |
| :------------: | :--------: |  ----- | ----- | :----: | :------: | --- |
| I2R-Net (Vanilla version, 1st stage:HRNet-W48-S) |  256x192  | 0.643 | 0.850 | 0.692  | [model](#) | [log](#) |
| I2R-Net (1st stage:TransPose-H) |  256x192  | 0.665 | 0.838 | 0.714  | [model](#) | [log](#) |
| I2R-Net (1st stage:HRFormer-B) |  256x192  | 0.678 | 0.850 | 0.728  | [model](#) | [log](#) |


### Results on COCO val2017 with detector

|     Model      | Input size |  AP    | Ap .5 | AP .75 | AP (M) | AP (L) |  AR   | AR (M) | AR (L) | Download | Log |
| :------------: | :--------: |  ----- | ----- | :----: | :----: | :----: | :---: | :----: | :----: | :------: | --- |
| I2R-Net (Vanilla version, 1st stage:HRNet-W48-S) |  256x192  | 0.753 | 0.902 | 0.819  | 0.717  | 0.824  | 0.805 | 0.761  | 0.868  | [model](#) | [log](#) |
| I2R-Net (1st stage:TransPose-H) |  256x192  | 0.758 | 0.904 | 0.821  | 0.720  | 0.829  | 0.809 | 0.766  | 0.873  | [model](#) | [log](#) |
| I2R-Net (1st stage:HRFormer-B) |  256x192  | 0.764 | 0.908 | 0.832  | 0.723  | 0.837  | 0.814 | 0.769  | 0.881  | [model](#) | [log](#) |
| I2R-Net (1st stage:HRFormer-B) |  384x288  | 0.773 | 0.910 | 0.836  | 0.730  | 0.845  | 0.821 | 0.777  | 0.886  | [model](#) | [log](#) |


We will update the codes and models soon.
