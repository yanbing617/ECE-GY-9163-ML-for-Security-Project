# ECE-GY-9163-ML-for-Security-Project

[TOC]

## 1.Background

​	可以有一些文献调研，或者复制ppd的一些东西

## 2.Fine Pruning Method

  prune的原理， 可以用 model_print的那个图

![model](D:\OneDrive - nyu.edu\_Fall2021\ML_Cyber\Assignment\lab3\CSAW-HackML-2020\lab3\model.png)

还有这种玩意.........

![](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013539684.png)



## 3.Dependencies and Data

### ***a.directories***

```bash
├── data  // Dataset is not uploaded due to poor network condition
    └── clean_valid_data.h5 // clean validation data used to prune and train the repaired network
    └── clean_test_data.h5  // clean test data used to evaluate the acc of the repaired network
    └── XXX_poisoned_data.h5  // clean test data used to evaluate the atk of the repaired network
├── models
    └── XXX_weights.h5 // weights to backdoored network
    └── XXX_bd_net.h5 // backdoored network with specific attack dataset
    └── XXX_bd_net_rp.h5 // pruned network generated from './prune.py'
└── prune.py  // generate a pruned network saved as './models/XXX_bd_net_rp.h5'
└── eval.py  // create a repaired network and eval the acc and atk on specific poisoned data
└── README.md  // Lab instruction and report
└── model.png  // Main structure of the backdoored network
```

### ***b.dependencies***

We use ***Anaconda*** environment and ***PyCharm*** to design this repaired network as a detector to defend against ***backdoor attack***, implementing **Fine-Pruning**, relating Python libararies are as follow:

      1. Python 3.7.9
      2. Keras 2.3.1
      3. Numpy 1.16.3
      4. Matplotlib 2.2.2
      5. H5py 2.9.0
      6. TensorFlow-gpu 1.15.2
      7. tensorflow-model-optimization 0.7.0

### ***c.data***

      1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
      2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
      3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.



## 4.Codes and Explanations

加一些说明（示例）

```python
  def on_train_begin(self, logs=None):
    # Collect all the prunable layers in the model.
    self.prunable_layers = pruning_wrapper.collect_prunable_layers(self.model)
    if not self.prunable_layers:
      return
    # If the model is newly created/initialized, set the 'pruning_step' to 0.
    # If the model is saved and then restored, do nothing.
    if self.prunable_layers[0].pruning_step == -1:
      tuples = []
      for layer in self.prunable_layers:
        tuples.append((layer.pruning_step, 0))
      K.batch_set_value(tuples)
```



## 5.Result

可以有可视化的图片，类似于

a.预测结果（示例）

对于图片，可以找**攻击成功**和**分类失败**的特别展示， 可以***胡乱分析***一下

![image-20211221011948201](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221011948201.png)

b.表格（acc和atk）

![image-20211221013650888](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013650888.png)

可以是top-misClassify 或者 top-atk

c.**网络性能展示**（个人认为老师喜欢看这种东西）

![image-20211221013742342](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013742342.png)

## 5.Running Instruction

To generate and evaluate the **repaired model**, execute `prune.py` by running:

```shell
python prune.py <bad_model_filename>
```

To generate and evaluate the **repaired model**, execute `eval.py` by running:

```shell
python eval.py <poisoned_data_filename> <bad_model_filename>
```

*E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.*

## 6.Conclusion

做个总结：结论是还不错

可以展望下其他方法

## 7.References

[1] Liu, Kang, Brendan Dolan-Gavitt, and Siddharth Garg. "Fine-pruning: Defending against backdooring attacks on deep neural networks." *International Symposium on Research in Attacks, Intrusions, and Defenses*. Springer, Cham, 2018.

[2] Gu, Tianyu, et al. "Badnets: Evaluating backdooring attacks on deep neural networks." *IEEE Access* 7 (2019): 47230-47244.

[3] ODSC Community. “What Is Pruning in Machine Learning?” *Open Data Science - Your News Source for AI, Machine Learning & More*, ODSC Community, 28 Oct. 2020, opendatascience.com/what-is-pruning-in-machine-learning/. 

[4]“Pruning in Keras Example  :  Tensorflow Model Optimization.” *TensorFlow*, Google, 11 Nov. 2021, www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras. 