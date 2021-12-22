# ECE-GY-9163-ML-for-Security-Project

## 0.Team members

We are a group of 3, members are as follow:

***Xinhao, LI***(xl3413)       xl3413@nyu.edu

***Yanbing, YANG***(yy3476)       yy3476@nyu.edu

***Zhe, ZHANG***(zz3230)       zz3230@nyu.edu

## 1.Background

In lab3, we have tried to prune channels in a max_pooling layer. However, this method does not work well.

In our project, we tried to do more pruning job. We believe that the backdoor in the model hide in the neurons which they are inactive when clean data is used. After doing some research, we found a way to prune the whole net, which is removing the individual weight connections from a network by setting them to 0. In this way, the connections will become no-operation in the network. This way of pruning is widely used in model compression, and it is also very useful in the problem we are facing.

In our code, we first set our sparsity to 50% and then increase to %70. In the end, 70% of the neurons will be disabled, which will slightly effect the model accuracy. However, we can prevent the attack.

Pruning is the process of removing weight connections in a network to increase inference speed and decrease model storage size. In general, neural networks are very over parameterized. Pruning a network basically means the defender finding and the backdoor neurons and eliminating the unused parameters. Because the neurons activated by the backdoor data are only activated by the backdoor, and they are dormant under the clean data, we record the average activation of each neuron. Then, the defender iteratively trims the neurons of the DNN in an increasing order of average activation, and records the trimming accuracy.



## 2.Fine Pruning Method

**network structure of defense method**

![model](https://github.com/yanbing617/ECE-GY-9163-ML-for-Security-Project/blob/main/model_architecture.png)

We will try to prune neurons in all **Dense layer**



## 3.Dependencies and Data

### ***a. directories***

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
└── model_architecture.png  // Main structure of the backdoored network
```

### ***b. dependencies***

We use ***Anaconda*** environment and ***PyCharm*** to design this repaired network as a detector to defend against ***backdoor attack***, implementing **Fine-Pruning**, relating Python libararies are as follow:

      1. Python 3.7.9
      2. Keras 2.3.1
      3. Numpy 1.16.3
      4. Matplotlib 2.2.2
      5. H5py 2.9.0
      6. TensorFlow-gpu 1.15.2
      7. tensorflow-model-optimization 0.7.0

### ***c. data***

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

#### a. Network Parameters after Pruning

![image-20211221013742342](C:\Users\soapi\AppData\Roaming\Typora\typora-user-images\image-20211221013742342.png)



#### b. Accuracy(test) and Success Rate(poisoned)

| BadNet           | Accuracy(original) | Accuracy(repaired) | Success Rate(original) | Success Rate(repaired) |
| ---------------- | ------------------ | ------------------ | ---------------------- | ---------------------- |
| sunglasses       | Content Cell       | Content Cell       | Content Cell           |                        |
| multiple_trigger | Content Cell       | Content Cell       | Content Cell           |                        |
| anonymous_1      | Content Cell       | Content Cell       | Content Cell           |                        |
| annoymous_2      | Content Cell       | Content Cell       | Content Cell           |                        |

As we can see, the attack success rate(ASR) has a **sharp drop** after implementing the fine-pruning method, mean while the drop of accuracy is acceptable

 

#### c. Misclassification and Defense Failure  

show 一些图片





## 6.Running Instruction

To train and generate the **pruned model**, execute `prune.py` by running:

```shell
python prune.py <bad_model_filename>
```

To generate and evaluate the **repaired model**, execute `eval.py` by running:

```shell
python eval.py <poisoned_data_filename> <bad_model_filename>
```

*E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.*



## 7.Conclusion

***The result is good***



## 8.References

[1] Liu, Kang, Brendan Dolan-Gavitt, and Siddharth Garg. "Fine-pruning: Defending against backdooring attacks on deep neural networks." *International Symposium on Research in Attacks, Intrusions, and Defenses*. Springer, Cham, 2018.

[2] Gu, Tianyu, et al. "Badnets: Evaluating backdooring attacks on deep neural networks." *IEEE Access* 7 (2019): 47230-47244.

[3] ODSC Community. “What Is Pruning in Machine Learning?” *Open Data Science - Your News Source for AI, Machine Learning & More*, ODSC Community, 28 Oct. 2020, opendatascience.com/what-is-pruning-in-machine-learning/. 

[4]“Pruning in Keras Example  :  Tensorflow Model Optimization.” *TensorFlow*, Google, 11 Nov. 2021, www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras. 