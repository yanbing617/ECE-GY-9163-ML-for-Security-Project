# ECE-GY-9163-ML-for-Security-Project

### Tips

1.**21日17:00前完成初级版本**

2. 提高完成度，多做图形代码和可视化展示

3.【建议】如果有时间的话可以多找些badNet或者自己训练下

4.注意markdown语法 github格式和typora的区别（有小区别）

5.方法为**fine pruning** ，做一个比较完整的过程

6. 代码大家都差不多，老师可能喜欢分析和思考更多
7. 以下所有的实际内容 **都是示例**， **不可作为最终结果**

[TOC]

### 1.Background

​	可以有一些文献调研，或者复制ppd的一些东西

### 2.原理

  prune的原理， 可以用 model_print的那个图

![model](D:\OneDrive - nyu.edu\_Fall2021\ML_Cyber\Assignment\lab3\CSAW-HackML-2020\lab3\model.png)

还有这种玩意.........

![](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013539684.png)



### 3.Dependencies and Data

###  运行环境（tensor_model_optimization可以省略，但写更好）

***a.目录格式及其说明***

```bash
├── data  // Dataset is not uploaded due to poor network condition
    └── cl
        └── valid.h5 // this is clean validation data used to design the defense
        └── test.h5  // this is clean test data used to evaluate the BadNet
    └── bd
        └── bd_valid.h5 // this is sunglasses poisoned validation data
        └── bd_test.h5  // this is sunglasses poisoned test data
├── models
    └── bd_net.h5
    └── bd_weights.h5
├── report //Pruned part of repaired network)
	└── pruning_channel_model_acc_decrease_by_2%.h5
    └── pruning_channel_model_acc_decrease_by_4%.h5
    └── pruning_channel_model_acc_decrease_by_10%.h5
    └── pruning_channel_model_acc_decrease_by_30%.h5  
├── detector.ipynb  // Source code
└── detector.pdf  // Results and figs of running Code
└── README.md  // Lab report
└── model.png  // Main structure of the backdoored network
└── ML_Security_.pdf  // Lab3 instruction and requirements
```

'pruning_channel_model_acc_decrease_by_2%.h5' 是XXXXXX

***c.依赖***

      1. Python 3.7.9
      2. Keras 2.3.1
      3. Numpy 1.16.3
      4. Matplotlib 2.2.2
      5. H5py 2.9.0
      6. TensorFlow-gpu 1.15.2

anaconda/ colab之类的

可以介绍下这个**tensorflow**和**tensor_model_optimization库**

***b.DATA***

google drive link

**【建议】如果用了 *自己找的* badNet和数据集，可以特别提一下**





### 4.Codes and Explanations

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



### 5.Result

可以有可视化的图片，类似于

a.预测结果（示例）

对于图片，可以找**攻击成功**和**分类失败**的特别展示， 可以***胡乱分析***一下

![image-20211221011948201](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221011948201.png)

b.表格（acc和atk）

![image-20211221013650888](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013650888.png)

可以是top-misClassify 或者 top-atk

c.**网络性能展示**（个人认为老师喜欢看这种东西）

![image-20211221013742342](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013742342.png)

参考文章

 [BadNets_Evaluating_Backdooring_Attacks_on_Deep_Neural_Networks.pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687) 

### 5.Running Instruction

示例

To evaluate the backdoored model, execute `eval.py` by running:  
`python3 eval.py <clean validation data directory> <model directory>`.

E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

### 6.Conclusion

做个总结：结论是还不错

可以展望下其他方法

### 7.References

[1] Liu, Kang, Brendan Dolan-Gavitt, and Siddharth Garg. "Fine-pruning: Defending against backdooring attacks on deep neural networks." *International Symposium on Research in Attacks, Intrusions, and Defenses*. Springer, Cham, 2018.

[2] Gu, Tianyu, et al. "Badnets: Evaluating backdooring attacks on deep neural networks." *IEEE Access* 7 (2019): 47230-47244.

[3] ODSC Community. “What Is Pruning in Machine Learning?” *Open Data Science - Your News Source for AI, Machine Learning & More*, ODSC Community, 28 Oct. 2020, opendatascience.com/what-is-pruning-in-machine-learning/. 