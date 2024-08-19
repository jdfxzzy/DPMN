## Improving Scene Text Image Super-Resolution via Dual Prior Modulation Network (AAAI 2023)

*[Shipeng Zhu](http://palm.seu.edu.cn/homepage/zhushipeng/demo/index.html), [Zuoyan Zhao](http://palm.seu.edu.cn/homepage/zhaozuoyan/index.html), [Pengfei Fang](https://fpfcjdsg.github.io/), [Hui Xue](http://palm.seu.edu.cn/hxue/)*

This repository offers the official Pytorch code for this paper. If you have any question, feel free to contact Zuoyan Zhao ([zuoyanzhao@seu.edu.cn](mailto:zuoyanzhao@seu.edu.cn)).

## Environment

![python](https://img.shields.io/badge/Python-v3.6-green.svg?style=plastic)  ![pytorch](https://img.shields.io/badge/Pytorch-v1.10-green.svg?style=plastic)  ![cuda](https://img.shields.io/badge/Cuda-v11.3-green.svg?style=plastic)  ![numpy](https://img.shields.io/badge/Numpy-v1.19-green.svg?style=plastic)

Other possible Python packages like PyYAML, pygame, Pillow and imgaug are also needed, please refer to *requirements.txt* for more information.

## Datasets and Pre-trained Recognizers

- Download the TextZoom dataset from: https://github.com/JasonBoy1/TextZoom.
- Download the pre-trained recognizers from:
  - ASTER: https://github.com/ayumiymk/aster.pytorch.
  - CRNN: https://github.com/meijieru/crnn.pytorch.
  - MORAN: https://github.com/Canjie-Luo/MORAN_v2.
  - VisionLAN: https://github.com/wangyuxin87/VisionLAN.
- **Notes:** 
  - It is necessary for you to modify the */config/super_resolution.yaml* file according to your path of dataset and recognizers.
  - If you use 3 VisionLANs to generate the graphic recognition results, please duplicate the pre-trained VisionLAN model for 3 times and rename them as *recognizer_best_0.pth*, *recognizer_best_1.pth*, *recognizer_best_2.pth* respectively. If you use other recognizers, please follow the same step.
  - It is also necessary for you to modify the */model/VisionLAN/cfgs/cfgs_eval.py* file according to your path of *dic_36.txt*, which is the dictionary of VisionLAN.

## Training and Testing the Model

- For example, if you want to train the model using TATT as PSN and ASTER as the corresponding recognizer, you can use the script:

  ```shell
  python main.py --arch="tatt" --batch_size="48" --STN --gradient --mask --vis_dir="./vis" --rec="aster" --patch_size="2,2,2,2,2,2," --embed_dim="96,96,96,96,96,96," --window_size="2,4,8,2,4,8,2,4,8,2,4,8,2,4,8,2,4,8," --mlp_ratio="4,4,4,4,4,4," --drop_rate="0.1,0.1,0.1,0.1,0.1,0.1," --attn_drop_rate="0.1,0.1,0.1,0.1,0.1,0.1," --drop_path_rate="0.1,0.1,0.1,0.1,0.1,0.1," --stu_iter_b1="3" --stu_iter_b2="3" --depths="1,1,1,1,1,1," --num_heads="6,6,6,6,6,6," --rec_path="/root/data1/recognizers" --font_path="./arial.ttf" --tpg="visionlan" --resume="/root/data1/TATT_ASTER" --rotate_train="5" --alpha="0.5"
  ```

  In this example, the two branch both have 3 PGRM blocks, so we have 6 PGRM blocks in total. In each PGRM, we set patch_size=2, embed_dim=96 and window_size=(2, 4, 8), then I think it is easy for you to understand the script above. You can set your own hyper-parameters, but please move the pre-trained recognizers to a dictionary (e.g. /root/data1/recognizers), and the pre-trained PSN to another dictionary (e.g. /root/data1/TATT_ASTER).

- For example, if you want to test the model using TATT as PSN and ASTER as the corresponding recognizer, you can use the script:

  ```shell
  python main.py --arch="tatt" --test --test_data_dir="/root/data1/datasets/TextZoom/test/easy" --batch_size="48" --gradient --mask --resume="/root/data1/DPMN/model_test" --rec="aster"  --patch_size="2,2,2,2,2,2," --embed_dim="96,96,96,96,96,96," --window_size="2,4,8,2,4,8,2,4,8,2,4,8,2,4,8,2,4,8," --mlp_ratio="4,4,4,4,4,4," --drop_rate="0.1,0.1,0.1,0.1,0.1,0.1," --attn_drop_rate="0.1,0.1,0.1,0.1,0.1,0.1," --drop_path_rate="0.1,0.1,0.1,0.1,0.1,0.1," --stu_iter_b1="3" --stu_iter_b2="3" --depths="1,1,1,1,1,1," --num_heads="6,6,6,6,6,6," --rec_path="/root/data1/DPMN/model_test" --font_path="./arial.ttf" --tpg="visionlan" --STN --alpha="0.5"
  ```

  It is necessary for you to move the PSN and the pre-trained DPMN to the same directory (e.g. /root/data1/DPMN/model_test), the fine-tuned recognizers for generating the graphic recognition results should also be moved to this directory.

## Pre-trained Weights of Our Implemented Models

- **Notes:** 
  - As is mentioned in our paper, our setting of experiments is different from previous works. Taking TATT as an example, they trained the model and used ASTER for evaluation to select the best model. Then it is clearly that this model is not the best if we use CRNN or MORAN for evaluation.
  - However, our DPMN aims to solve the drawbacks of PSNs, as is thoroughly clarified in our paper. So it is more meaningful for us to use the best TATT selected by CRNN as the PSN if we use TATT as PSN and CRNN as the recognizer for evaluation.
  - For simplicity, we use the notation TATT-ASTER to represent the best TATT model selected by ASTER. TATT-CRNN, TATT-MORAN, etc. are also similar.
  - Based on the requirement above, we need 15 PSNs and then use ASTER, CRNN and MORAN respectively to train our DPMN. Among these PSNs, TSRN-ASTER is provided by [@yustiks](https://github.com/yustiks), as you can see in [this issue](https://github.com/JasonBoy1/TextZoom/issues/8#issuecomment-767552860). [TBSRN-CRNN](https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope), [TG-CRNN](https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt) and [TATT-ASTER](https://github.com/mjq11302010044/TATT) are provided by their authors respectively, you can download them from the corresponding github repository. Other PSNs are our trained version.
- Considering that TATT is the SOTA STISR method with open source code when we submit our paper to AAAI, we release our implemented DPMN for TATT-ASTER, TATT-CRNN and TATT-MORAN now to guarantee reproducibility. You can download them from the following links. Our zip file also contains the corresponding PSN and script for testing. However, as is mentioned above, TATT-ASTER is NOT our implemented version and its author didn't made it  public. So we didn't provided the pre-trained weight of TATT-ASTER in our zip file according to our promise to its author. If you really need this model, please follow [this issue](https://github.com/mjq11302010044/TATT/issues/4#issuecomment-1100872209).
  - DPMN with TATT-ASTER as PSN:
       - Baidu Netdisk: https://pan.baidu.com/s/1grBb_RlR1QMCgs54dW4bfw, password: jtei.
       - Google Drive: https://drive.google.com/file/d/1keM__l4aj26b24pq85UPkUKhZjS6hdXq/view?usp=sharing
  - DPMN with TATT-CRNN as PSN:
       - Baidu Netdisk: https://pan.baidu.com/s/1e0Qiq21vNH0Xq2kLkwQuig, password: 8r6g.
       - Google Drive: https://drive.google.com/file/d/1226-SZ6YAsZh_a5-5GmGEvQn51ntDskj/view?usp=sharing
  - DPMN with TATT-MORAN as PSN:
       - Baidu Netdisk: https://pan.baidu.com/s/1ch_UHrupjuCs5eaiWfE02Q, password: 8hrz.
       - Google Drive: https://drive.google.com/file/d/1K23n0xuzLhbsKxOLzBmzA9tDXWs2q5it/view?usp=sharing

## Frequently Asked Questions

- **Q:** Why does the results of TSRN / TBSRN / TG / TPGSR / TATT in Table 1 different from what their paper reported?

  **A:** We have partly explained the reason in the notes of "Pre-trained Weights of Our Implemented Models" section. The recognition accuracy of the models released by the authors are not identical to what their paper reported, let alone our implemented version. Considering that we use these models as PSN and use our DPMN to refine their SR results, we reported the recognition accuracy of the pre-trained models instead of what their paper reported for the sake of fairness.

- **Q:** Why does the recognition accuracy of HR images in Table 1 different from what the [ECCV 2020](https://arxiv.org/abs/2005.03341) paper reported?

  **A:** This issue has something to do with the version of Pytorch. If we use Pytorch 1.2, we will get the same result as what the [ECCV 2020](https://arxiv.org/abs/2005.03341) paper reported. However, TATT was implemented by Pytorch 1.8. If we use such a lower version, it is impossible for us to load the pre-trained weights. Thus we use a newer version, Pytorch 1.10 for simplicity and reported the recognition accuracy under this version of Pytorch for preciseness.

## Acknowledgement

- We inherited most of the frameworks from [TATT](https://github.com/mjq11302010044/TATT) and use the pre-trained [ASTER](https://github.com/ayumiymk/aster.pytorch), [CRNN](https://github.com/meijieru/crnn.pytorch), [MORAN](https://github.com/Canjie-Luo/MORAN_v2) and [VisionLAN](https://github.com/wangyuxin87/VisionLAN) model for recognition. Thank you for your contribution!
- Some of our code are modified from [DW-ViT](https://github.com/pzhren/DW-ViT), [MMFL](https://github.com/AliceQLin/MMFL-Inpainting) and [SRNet-Datagen](https://github.com/youdao-ai/SRNet-Datagen). Thank you for your contribution!
- Thank you to the author of TBSRN, TG and TATT for making their code and pre-trained weights public.

## Citation

```
@inproceedings{zhu2023improving,
  title={Improving Scene Text Image Super-Resolution via Dual Prior Modulation Network},
  author={Shipeng Zhu and Zuoyan Zhao and Pengfei Fang and Hui Xue},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={3843--3851},
  year={2023}
}
```
