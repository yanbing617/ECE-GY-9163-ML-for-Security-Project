import tensorflow as tf
import h5py
import numpy as np
import tensorflow_model_optimization as tfmot
import sys

clean_data_filename = './data/clean_validation_data.h5'
clean_test_filename = './data/clean_test_data.h5'
model_filename = str(sys.argv[1])
output_filename = model_filename.split('.')[0] + '_rp.h5'

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data / 255.0, y_data

def prune_model(base_model, initial_sparsity, final_sparsity, end_step, log_dir):
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                                 final_sparsity=final_sparsity,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    model_for_pruning.compile(optimizer=base_model.optimizer,
                              loss=base_model.loss,
                              metrics=['accuracy'])
    log_dir = 'log'
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
    ]
    return model_for_pruning, callbacks



if __name__ == "__main__":
    #Congiguration for GPU
    TF_CONFIG_ = tf.compat.v1.ConfigProto()
    TF_CONFIG_.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config = TF_CONFIG_)
    # load data and model
    x_test, y_test = data_loader(clean_test_filename)
    x_train, y_train = data_loader(clean_data_filename)
    bd_model = tf.keras.models.load_model(model_filename)
    base_model = tf.keras.models.load_model(model_filename)
    # Hyper param
    batch_size = 10
    epochs = 2
    validation_split = 0.1
    # compute end_step
    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    # prune
    pruned_model, callbacks = prune_model(base_model, 0.5, 0.7, end_step, "log")
    # fit
    pruned_model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                        callbacks=callbacks)
    # save the model
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    tf.keras.models.save_model(model_for_export, output_filename, include_optimizer=False)
    print('Saved pruned Keras model to:', output_filename)