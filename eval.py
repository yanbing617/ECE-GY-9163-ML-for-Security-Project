import keras
import sys
import h5py
import numpy as np
import tensorflow_model_optimization as tfmot

clean_data_filename = "./data/clean_test_data.h5"
poisoned_data_filename = str(sys.argv[1])
bd_model_filename = str(sys.argv[2])
rp_model_filename = bd_model_filename.split('.')[0] + '_rp.h5'

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data / 255.0, y_data


class RepairModel:
    def __init__(self, bd_model, rp_model, class_cnt):
        self.bd_model = bd_model
        self.rp_model = rp_model
        self.class_cnt = class_cnt

    def predict(self, x_test):
        y_bd = np.argmax(self.bd_model.predict(x_test), axis=1)
        y_rp = np.argmax(self.rp_model.predict(x_test), axis=1)
        res = []
        for i in range(len(y_bd)):
            if y_bd[i] == y_rp[i]:
                res.append(y_bd[i])
            else:
                res.append(self.class_cnt)
        return np.array(res)


def main():
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    bd_model = keras.models.load_model(bd_model_filename)
    rp_model = keras.models.load_model(rp_model_filename)

    class_cnt = bd_model.output.shape[1]

    good_net = RepairModel(bd_model, rp_model, class_cnt)

    cl_label_p = good_net.predict(cl_x_test)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print('Clean Classification accuracy:', clean_accuracy)

    bd_label_p = good_net.predict(bd_x_test)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate:', asr)


if __name__ == '__main__':
    main()
