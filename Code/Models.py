import math, re, gc
import numpy as np # linear algebra
import pickle
from datetime import datetime, timedelta
import tensorflow as tf
import efficientnet.tfkeras as efficientnet
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print('TensorFlow version', tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

training_dataset = get_training_dataset()
training_dataset = training_dataset.unbatch().batch(20)
train_batch = iter(training_dataset)
#
#display_batch_of_images(next(train_batch))

test_dataset = get_test_dataset()
test_dataset = test_dataset.unbatch().batch(20)
test_batch = iter(test_dataset)
#
#display_batch_of_images(next(test_batch))
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True)


class CosLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CosLayer, self).__init__()
        self.units = units

    #         self.w = self.add_weight('kernel', shape=(2560, self.units),
    #                                  initializer='random_normal',
    #                                  trainable=True)

    def build(self, input_shape):
        self.w = self.add_weight('kernel', shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        w = tf.math.l2_normalize(self.w, axis=0, name='weight_normalization')
        inputs = tf.math.l2_normalize(inputs, axis=-1, name='feature_normalization')
        #         a_w = tf.sqrt(tf.reduce_sum(self.w ** 2, axis=, keepdims))
        outputs = tf.matmul(inputs, w)
        return tf.clip_by_value(outputs, -1, 1)


def am_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(tf.squeeze(y_true, axis=-1), len(CLASSES))
    y_pred = tf.exp(10 * (y_pred - 0.3 * y_true))

    numerator = tf.reduce_sum(y_pred * y_true, axis=-1)
    denominator = tf.reduce_sum(y_pred, axis=-1)
    am_loss = -tf.math.log(numerator / (denominator + 1e-8))

    return tf.reduce_mean(am_loss)


def create_VGG16_model():
    pretrained_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True # False

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model

def create_VGG16_model_am_loss():
    pretrained_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True # False

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        CosLayer(len(CLASSES))
    ])

    model.compile(optimizer = 'adam', loss = am_loss, metrics = ['sparse_categorical_accuracy'])
    return model


def create_Xception_model():
    pretrained_model = tf.keras.applications.Xception(include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model


def create_Xception_model_am_loss():
    pretrained_model = tf.keras.applications.Xception(include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        CosLayer(len(CLASSES))
    ])

    model.compile(optimizer = 'adam', loss = am_loss, metrics = ['sparse_categorical_accuracy'])
    return model

def create_DenseNet_model():
    pretrained_model = tf.keras.applications.DenseNet201(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model

def create_DenseNet_model_am_loss():
    pretrained_model = tf.keras.applications.DenseNet201(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        CosLayer(len(CLASSES))
    ])

    model.compile(optimizer = 'adam', loss = am_loss, metrics = ['sparse_categorical_accuracy'])
    return model

def create_EfficientNet_model():
    pretrained_model = efficientnet.EfficientNetB7(weights = 'noisy-student', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model

def create_EfficientNet_model_am_loss():
    pretrained_model = efficientnet.EfficientNetB7(weights = 'noisy-student', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        CosLayer(len(CLASSES))
    ])

    model.compile(optimizer = 'adam', loss = am_loss, metrics = ['sparse_categorical_accuracy'])
    return model

def create_InceptionV3_model():
    pretrained_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model


def create_InceptionV3_model_am_loss():
    pretrained_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        CosLayer(len(CLASSES))
    ])

    model.compile(optimizer = 'adam', loss = am_loss, metrics = ['sparse_categorical_accuracy'])
    return model

def create_ResNet152_model():
    pretrained_model = tf.keras.applications.ResNet152V2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model

def create_MobileNetV2_model():
    pretrained_model = tf.keras.applications.MobileNetV2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model

def create_InceptionResNetV2_model():
    pretrained_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
    return model

def create_InceptionResNetV2_model_am_loss():
    pretrained_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        CosLayer(len(CLASSES))
    ])

    model.compile(optimizer = 'adam', loss = am_loss, metrics = ['sparse_categorical_accuracy'])
    return model

no_of_models = 1
models = [0] * no_of_models
start_model = 0
end_model = 1
model_indx_0 = start_model
#model_indx_1 = start_model + 1
#model_indx_2 = start_model + 2

val_probabilities = [0] * no_of_models
test_probabilities = [0] * no_of_models
all_probabilities = [0] * no_of_models

#with strategy.scope():
#    models[0] = create_DenseNet_model()
#    models[1] = create_EfficientNet_model()
#print(models[0].summary())
#print(models[1].summary())
#
with strategy.scope():
    for j in range(no_of_models):
#        models[j] = create_VGG16_model()
#        models[j] = create_Xception_model()
#        models[j] = create_DenseNet_model()
#        models[j] = create_EfficientNet_model()
         models[j] = create_InceptionV3_model()
#        models[j] = create_ResNet152_model()
#        models[j] = create_MobileNetV2_model()
#        models[j] = create_InceptionResNetV2_model()
#        models[j] = create_Xception_model_am_loss()
#        models[j] = create_DenseNet_model_am_loss()
#        models[j] = create_EfficientNet_model_am_loss()
#        models[j] = create_InceptionV3_model_am_loss()
#        models[j] = create_ResNet152_model()
#        models[j] = create_MobileNetV2_model()
#        models[j] = create_InceptionResNetV2_model_am_loss()

models[0].summary()

def write_history(j):
    history_dict = [0] * no_of_models
    for i in range(j + 1):
        if (historys[i] != 0):
            history_dict[i] = historys[i].history
#
    filename = './' + this_run_file_prefix + 'model_history_' + str(j) + '.pkl'
    pklfile = open(filename, 'ab')
    pickle.dump(history_dict, pklfile)
    pklfile.close()

EPOCHS = 50 # 35 # 2 # 20
historys = [0] * no_of_models
#lr_exp_decay_values = [0.5,0.6,0.5,0.7] # [0.6,0.7,0.8,0.9,0.6,0.7,0.8,0.9,0.6,0.7,0.8,0.9,0.6,0.7,0.8,0.9,0.5,0.5,0.5,0.5]
#lr_max_values = [0.00005,0.00003,0.00004,0.00003] # [0.00003,0.00003,0.00003,0.00003,0.00004,0.00004,0.00004,0.00004,0.00005,0.00005,0.00005,0.00005,0.00006,0.00006,0.00006,0.00006,0.00003,0.00004,0.00005,0.00006]
finished_models = 0

for j in range(start_model, end_model):
    start_training = datetime.now()
    print(start_training)
    time_from_start_program_tdelta = start_training - start_time
    if time_from_start_program_tdelta > end_training_by_tdelta:
        print(j, 'time limit for doing training over, get out')
        break
#    with strategy.scope():
#        models[j] = create_DenseNet_model()
#    if j == 0:
#        models[0].summary()
#        print('----------------------------------------------------')
#    LR_EXP_DECAY = lr_exp_decay_values[j]
#    LR_MAX = lr_max_values[j] * strategy.num_replicas_in_sync
    print('LR_EXP_DECAY:', LR_EXP_DECAY, '. LR_MAX:', LR_MAX)
    historys[j] = models[j].fit(get_training_dataset(), steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, validation_data = get_validation_dataset(), callbacks = [lr_callback, early_stop])
    write_history(j)
    filename = this_run_file_prefix + 'models_' + str(j) + '.h5'
    models[j].save(filename)
#    model_to_delete = models[j]
#    models[j] = 0
#    del model_to_delete
    gc.collect()
    finished_models = j + 1

print(datetime.now())
#
cmdataset = get_validation_dataset(ordered = True)
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()
cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

test_ds = get_test_dataset(ordered = True)

#print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
#
dataset = get_validation_dataset()
dataset = dataset.unbatch().batch(20)
batch = iter(dataset)

images, labels = next(batch)

print(datetime.now())
#
for j in range(start_model, end_model):
    val_probabilities[j] = models[j].predict(images_ds)
    test_probabilities[j] = models[j].predict(test_images_ds)
    all_probabilities[j] = models[j].predict(images)

print(datetime.now())
#
for j in range(start_model, finished_models):
    display_training_curves(historys[j].history['loss'], historys[j].history['val_loss'], 'loss', 211)
    display_training_curves(historys[j].history['sparse_categorical_accuracy'], historys[j].history['val_sparse_categorical_accuracy'], 'accuracy', 212)
#
for j in range(start_model, finished_models):
    print('model number:', j, ', Train Accuracy:', max(historys[j].history['sparse_categorical_accuracy']), ', Validation Accuracy:', max(historys[j].history['val_sparse_categorical_accuracy']))
for j in range(start_model, finished_models):
    print('model number:', j, ', Train Loss:', min(historys[j].history['loss']), ', Validation Loss:', min(historys[j].history['val_loss']))
#
cm_probabilities = np.zeros((val_probabilities[0].shape)) # = val_probabilities[0] + val_probabilities[1] + val_probabilities[2]
for j in range(no_of_models):
    cm_probabilities = cm_probabilities + val_probabilities[j]

cm_predictions = np.argmax(cm_probabilities, axis = -1)
print('Correct labels: ', cm_correct_labels.shape, cm_correct_labels)
print('Predicted labels: ', cm_predictions.shape, cm_predictions)

def getFitPrecisionRecall(correct_labels, predictions):
    score = f1_score(correct_labels, predictions, labels = range(len(CLASSES)), average = 'macro')
    precision = precision_score(correct_labels, predictions, labels = range(len(CLASSES)), average = 'macro')
    recall = recall_score(correct_labels, predictions, labels = range(len(CLASSES)), average = 'macro')
    return score, precision, recall
#
cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels = range(len(CLASSES)))
score, precision, recall = getFitPrecisionRecall(cm_correct_labels, cm_predictions)
cmat = (cmat.T / cmat.sum(axis = -1)).T
display_confusion_matrix(cmat, score, precision, recall)
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

def create_submission_file(filename, probabilities):
    predictions = np.argmax(probabilities, axis = -1)
    print('Generating submission file...', filename)
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

    np.savetxt(filename, np.rec.fromarrays([test_ids, predictions]), fmt = ['%s', '%d'], delimiter = ',', header = 'id,label', comments = '')
#

probabilities = np.zeros((test_probabilities[0].shape)) # = test_probabilities[0] + test_probabilities[1] + test_probabilities[2]
for j in range(no_of_models):
    probabilities = probabilities + test_probabilities[j]

filename = this_run_file_prefix + 'submission.csv'
create_submission_file(filename, probabilities)
create_submission_file('submission.csv', probabilities)
#

def combine_two(correct_labels, probability_0, probability_1):
    print('Start. ', datetime.now())
    alphas0_to_try = np.linspace(0, 1, 101)
    best_score = -1
    best_alpha0 = -1
    best_alpha1 = -1
    best_precision = -1
    best_recall = -1
    best_val_predictions = None

    for alpha0 in alphas0_to_try:
        alpha1 = 1.0 - alpha0
        probabilities = alpha0 * probability_0 + alpha1 * probability_1 #
        predictions = np.argmax(probabilities, axis = -1)

        score, precision, recall = getFitPrecisionRecall(correct_labels, predictions)
        if score > best_score:
            best_alpha0 = alpha0
            best_alpha1 = alpha1
            best_score = score
            best_precision = precision
            best_recall = recall
            best_val_predictions = predictions
    #
    return best_alpha0, best_alpha1, best_val_predictions, best_score, best_precision, best_recall

def combine_three(correct_labels, probability_0, probability_1, probability_2):
    print('Start. ', datetime.now())
    alphas0_to_try = np.linspace(0, 1, 101)
    alphas1_to_try = np.linspace(0, 1, 101)
    best_score = -1
    best_alpha0 = -1
    best_alpha1 = -1
    best_alpha2 = -1
    best_precision = -1
    best_recall = -1
    best_val_predictions = None

    for alpha0 in alphas0_to_try:
        for alpha1 in alphas1_to_try:
            if (alpha0 + alpha1) > 1.0:
                break

            alpha2 = 1.0 - alpha0 - alpha1
            probabilities = alpha0 * probability_0 + alpha1 * probability_1 + alpha2 * probability_2
            predictions = np.argmax(probabilities, axis = -1)

            score, precision, recall = getFitPrecisionRecall(correct_labels, predictions)
            if score > best_score:
                best_alpha0 = alpha0
                best_alpha1 = alpha1
                best_alpha2 = alpha2
                best_score = score
                best_precision = precision
                best_recall = recall
                best_val_predictions = predictions
    #
    return best_alpha0, best_alpha1, best_alpha2, best_val_predictions, best_score, best_precision, best_recall

def get_best_combination(no_models, cm_correct_labels, val_probabilities, test_probabilities):
    best_fit_score = -10000.0
    best_predictions = 0
    choose_filename = ''

    curr_predictions = np.argmax(val_probabilities[0], axis = -1)
    score, precision, recall = getFitPrecisionRecall(cm_correct_labels, curr_predictions)
    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
    filename = this_run_file_prefix + 'submission_0.csv'
    if best_fit_score < score:
        best_fit_score = score
        best_predictions = curr_predictions
        choose_filename = filename
        create_submission_file('./submission.csv', test_probabilities[0])
    create_submission_file(filename, test_probabilities[0])

    if no_models > 1:
        curr_predictions = np.argmax(val_probabilities[1], axis = -1)
        score, precision, recall = getFitPrecisionRecall(cm_correct_labels, curr_predictions)
        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
        filename = this_run_file_prefix + 'submission_1.csv'
        if best_fit_score < score:
            best_fit_score = score
            best_predictions = curr_predictions
            choose_filename = filename
            create_submission_file('./submission.csv', test_probabilities[1])
        create_submission_file(filename, test_probabilities[1])

    if no_models > 2:
        curr_predictions = np.argmax(val_probabilities[2], axis = -1)
        score, precision, recall = getFitPrecisionRecall(cm_correct_labels, curr_predictions)
        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
        filename = this_run_file_prefix + 'submission_2.csv'
        if best_fit_score < score:
            best_fit_score = score
            best_predictions = curr_predictions
            choose_filename = filename
            create_submission_file('./submission.csv', test_probabilities[2])
        create_submission_file(filename, test_probabilities[2])

    if no_models > 1:
        best_alpha0, best_alpha1, best_val_predictions, best_score, best_precision, best_recall = combine_two(cm_correct_labels, val_probabilities[0], val_probabilities[1])
        print('For indx', [0, 1], 'best_alpha0:', best_alpha0, 'best_alpha1:', best_alpha1, '. ', datetime.now())
        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(best_score, best_precision, best_recall))
        combined_probabilities = best_alpha0 * test_probabilities[0] + best_alpha1 * test_probabilities[1]
        filename = this_run_file_prefix + 'submission_01.csv'
        if best_fit_score < best_score:
            best_fit_score = best_score
            best_predictions = best_val_predictions
            choose_filename = filename
            create_submission_file('./submission.csv', combined_probabilities)
        create_submission_file(filename, combined_probabilities)

    if no_models > 2:
        best_alpha0, best_alpha1, best_val_predictions, best_score, best_precision, best_recall = combine_two(cm_correct_labels, val_probabilities[0], val_probabilities[2])
        print('For indx', [0, 2], 'best_alpha0:', best_alpha0, 'best_alpha1:', best_alpha1, '. ', datetime.now())
        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(best_score, best_precision, best_recall))
        combined_probabilities = best_alpha0 * test_probabilities[0] + best_alpha1 * test_probabilities[2]
        filename = this_run_file_prefix + 'submission_02.csv'
        if best_fit_score < best_score:
            best_fit_score = best_score
            best_predictions = best_val_predictions
            choose_filename = filename
            create_submission_file('./submission.csv', combined_probabilities)
        create_submission_file(filename, combined_probabilities)

        best_alpha0, best_alpha1, best_val_predictions, best_score, best_precision, best_recall = combine_two(cm_correct_labels, val_probabilities[1], val_probabilities[2])
        print('For indx', [1, 2], 'best_alpha0:', best_alpha0, 'best_alpha1:', best_alpha1, '. ', datetime.now())
        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(best_score, best_precision, best_recall))
        combined_probabilities = best_alpha0 * test_probabilities[1] + best_alpha1 * test_probabilities[2]
        filename = this_run_file_prefix + 'submission_12.csv'
        if best_fit_score < best_score:
            best_fit_score = best_score
            best_predictions = best_val_predictions
            choose_filename = filename
            create_submission_file('./submission.csv', combined_probabilities)
        create_submission_file(filename, combined_probabilities)

        best_alpha0, best_alpha1, best_alpha2, best_val_predictions, best_score, best_precision, best_recall = combine_three(cm_correct_labels, val_probabilities[0], val_probabilities[1], val_probabilities[2])
        print('For indx', [0, 1, 2], 'best_alpha0:', best_alpha0, 'best_alpha1:', best_alpha1, 'best_alpha2:', best_alpha2, '. ', datetime.now())
        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(best_score, best_precision, best_recall))
        combined_probabilities = best_alpha0 * test_probabilities[0] + best_alpha1 * test_probabilities[1] + best_alpha2 * test_probabilities[2]
        filename = this_run_file_prefix + 'submission_012.csv'
        if best_fit_score < best_score:
            best_fit_score = best_score
            best_predictions = best_val_predictions
            choose_filename = filename
            create_submission_file('./submission.csv', combined_probabilities)
        create_submission_file(filename, combined_probabilities)
#
    cmat = confusion_matrix(cm_correct_labels, best_predictions, labels = range(len(CLASSES)))
    cmat = (cmat.T / cmat.sum(axis = -1)).T
    display_confusion_matrix(cmat, best_fit_score, precision, recall)
#
    print('Best score from all combination was', best_fit_score, '. For submission file used is', choose_filename)
    return best_predictions
#
best_predictions = cm_predictions
if no_of_models > 1:
#    bp = get_best_combination(no_of_models, cm_correct_labels_results[0], val_probabilities, test_probabilities)
    bp = get_best_combination(no_of_models, cm_correct_labels, val_probabilities, test_probabilities)
    best_predictions = bp
#
probabilities = np.zeros((all_probabilities[0].shape)) # = all_probabilities[0] + all_probabilities[1] + all_probabilities[2]
for j in range(no_of_models):
    probabilities = probabilities + all_probabilities[j]

predictions = np.argmax(probabilities, axis =-1)
display_batch_of_images((images, labels), predictions)

# Storing the Output for Ensembling

#val_probs = [cm_correct_labels, cm_predictions, test_ids, val_probabilities[0], val_probabilities[1], val_probabilities[2], test_probabilities[0], test_probabilities[1], test_probabilities[2]]
#val_probs = [cm_correct_labels, cm_predictions, test_ids, val_probabilities[0], val_probabilities[1], test_probabilities[0], test_probabilities[1]]
val_probs = [cm_correct_labels, cm_predictions, test_ids, val_probabilities[0], test_probabilities[model_indx_0]]
filename = this_run_file_prefix + 'tests_vals_0.pkl'
pklfile = open(filename, 'ab')
pickle.dump(val_probs, pklfile)
pklfile.close()

#images_ds_unbatched = images_ds.unbatch()
#cm_images_ds_numpy = next(iter(images_ds_unbatched.batch(NUM_VALIDATION_IMAGES))).numpy()
use_correct_labels = cm_correct_labels
use_val_predictions = best_predictions

print('type of labels_ds is {}'.format(type(labels_ds)))
print('type of use_val_predictions is {}. shape of use_val_predictions is {}'.format(type(use_val_predictions), use_val_predictions.shape))
#print('type of use_correct_labels is {}, cm_images_ds_numpy is {}'.format(type(use_correct_labels), type(cm_images_ds_numpy)))
#print('shape of use_correct_labels is {}, cm_images_ds_numpy is {}'.format(use_correct_labels.shape, cm_images_ds_numpy.shape))

correct_labels_cnt = 0
incorrect_labels_cnt = 0
correct_labels = []
incorrect_labels = []
vals_actual_true = {}
vals_tp = {}
vals_fn = {}
vals_fp = {}
for i in range(len(CLASSES)):
    vals_actual_true[i] = 0
    vals_tp[i] = 0
    vals_fn[i] = 0
    vals_fp[i] = 0

for i in range(len(use_correct_labels)):
    correct_label = use_correct_labels[i]
    predict_label = use_val_predictions[i]
    vals_actual_true[correct_label] = vals_actual_true[correct_label] + 1
    if use_val_predictions[i] != use_correct_labels[i]:
        incorrect_labels_cnt = incorrect_labels_cnt + 1
        incorrect_labels.append(i)
        vals_fn[correct_label] = vals_fn[correct_label] + 1
        vals_fp[predict_label] = vals_fp[predict_label] + 1
    else:
        correct_labels_cnt = correct_labels_cnt + 1
        correct_labels.append(i)
        vals_tp[correct_label] = vals_tp[correct_label] + 1
#        print(i)
#
print('Number of correct_labels is {}, incorrect_labels is {}'.format(correct_labels_cnt, incorrect_labels_cnt))
#print('Correct labels', correct_labels)
print('Incorrect labels', incorrect_labels)
#
def display_my_batch_of_images(databatch, rows = 0, cols = 0, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = databatch
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    if rows == 0 or cols == 0:
        rows = int(math.sqrt(len(images)))
        cols = (len(images) + rows - 1)//rows
    print('Total number of images is {}, rows is {}, cols is {}'.format(len(images), rows, cols))

    # size and spacing
    FIGSIZE = 20.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
#
#disp_images = []
disp_labels = []
disp_predictions = []
for i in range(54):
    if i >= incorrect_labels_cnt:
        break
    id = incorrect_labels[i]
    disp_labels.append(use_correct_labels[id])
    disp_predictions.append(use_val_predictions[id])
#    disp_images.append(cm_images_ds_numpy[id])
#
print(disp_labels)
print(disp_predictions)
#
val_ids = list(range(len(use_correct_labels)))
filename = this_run_file_prefix + 'validation_results.csv'
np.savetxt(filename, np.rec.fromarrays([val_ids, use_correct_labels, use_val_predictions]), fmt = ['%d', '%d', '%d'], delimiter = ',', header = 'id,correct_label,predicted_label', comments = '')
#
cls_ids = list(range(len(CLASSES)))
#print(len(cls_ids), len(vals_actual_true), len(vals_tp), len(vals_fn), len(vals_fp))
filename = this_run_file_prefix + 'validation_statistics.csv'
np.savetxt(filename, np.rec.fromarrays([cls_ids, list(vals_actual_true.values()), list(vals_tp.values()), list(vals_fn.values()), list(vals_fp.values())]), fmt = ['%d', '%d', '%d', '%d', '%d'], delimiter = ',', header = 'cls_id,actual_true,true_positive,false_negative,false_positive', comments = '')
#