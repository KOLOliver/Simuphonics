import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from multiprocessing import Queue, Process, Pipe 
import time 
import gc

#Globals
sample_rate = 44100
length_in_seconds = 1
total_length = 44100
gen_num = 5
normal_sample = 100

# preprocess, shuffle and batch
def preprocess(wav): 
    global total_length, length_in_seconds
    samplerate, data = wavfile.read(wav)  # read wav data
    if(type(data[0]) ==  type(np.array([0]))):
        data = [np.float32(datatup[0])/32767 for datatup in data]
    else:
        data = [np.float32(d)/32767 for d in data]
    return data

#Create the dataset
def createDataset(label):
    pos_data = np.array([preprocess(f'{label}.wav') for i in range(gen_num)])
    pos_label = np.array([np.array([1]) for i in range(gen_num)])
    return pos_data, pos_label

def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random.uniform([], minval=-rad, maxval=rad+1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

def create_generator(gen):
    # gen = tf.keras.Sequential()
    gen.add(tf.keras.layers.Flatten(input_shape=(normal_sample,1)))

    # gen.add(tf.keras.layers.Dense(128, activation='tanh', use_bias=True))
    # gen.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # gen.add(tf.keras.layers.Lambda(lambda x: tf.concat([x[:-1000]-0.001*x[1000:],x[-1000:]], 0))) #comb filter
    gen.add(tf.keras.layers.Dense(10 * 10 * 9 * 7 * 7, activation="tanh", use_bias=True))
    # gen.add(tf.keras.layers.BatchNormalization())
    gen.add(tf.keras.layers.Lambda(lambda x: x * tf.math.sigmoid(x))) #swish function

    gen.add(tf.keras.layers.Reshape((10, 4410)))

    gen.add(tf.keras.layers.Conv1DTranspose(441, strides=10, kernel_size=25, padding="same"))
    # gen.add(tf.keras.layers.BatchNormalization())
    # gen.add(tf.keras.layers.ReLU())
    gen.add(tf.keras.layers.Lambda(lambda x: x * tf.math.sigmoid(x)))

    gen.add(tf.keras.layers.Conv1DTranspose(49, strides=9, kernel_size=25, padding="same"))
    # gen.add(tf.keras.layers.BatchNormalization())
    # gen.add(tf.keras.layers.ReLU())
    gen.add(tf.keras.layers.Lambda(lambda x: x * tf.math.sigmoid(x)))

    gen.add(tf.keras.layers.Conv1DTranspose(7, strides=7, kernel_size=25, padding="same"))
    # gen.add(tf.keras.layers.BatchNormalization())
    # gen.add(tf.keras.layers.ReLU())
    gen.add(tf.keras.layers.Lambda(lambda x: x * tf.math.sigmoid(x)))

    gen.add(tf.keras.layers.Conv1DTranspose(1, strides=7, kernel_size=25, padding="same"))
    gen.add(tf.keras.layers.Lambda(lambda x: tf.math.tanh(x)))
    # gen.add(tf.keras.layers.Lambda(lambda x: tf.concat([x[:-1000]-0.001*x[1000:],x[-1000:]], 0))) #comb filter
    # gen.add(tf.keras.layers.Lambda(lambda x: x/tf.math.reduce_max(x))) #scaling normalization

    # gen.add(tf.keras.layers.Reshape((44100, 1)))
    # gen.add(tf.keras.layers.Conv1D(1, 3, activation='tanh', padding="same"))
    # gen.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    # return gen

def create_discriminator(model):
    # model = Sequential()

    model.add(tf.keras.layers.ZeroPadding1D(padding=1,input_shape=(total_length,1)))
    model.add(tf.keras.layers.Conv1D(1, 3, strides=1, activation='linear', input_shape=(total_length,1), name='convlayer'))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Lambda(lambda x: apply_phaseshuffle(x, 1)))

    model.add(tf.keras.layers.ZeroPadding1D(padding=1,input_shape=(total_length,1)))
    model.add(tf.keras.layers.Conv1D(1, 3, strides=1, activation=tf.keras.activations.relu, input_shape=(total_length,1), name='conv2layer'))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Lambda(lambda x: apply_phaseshuffle(x, 1)))

    model.add(tf.keras.layers.ZeroPadding1D(padding=1,input_shape=(total_length,1)))
    model.add(tf.keras.layers.Conv1D(1, 3, strides=1, activation=tf.keras.activations.relu, input_shape=(total_length,1), name='conv3layer'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Lambda(lambda x: apply_phaseshuffle(x, 1)))

    model.add(tf.keras.layers.ZeroPadding1D(padding=1,input_shape=(total_length,1)))
    model.add(tf.keras.layers.Conv1D(1, 3, strides=1, activation=tf.keras.activations.relu, input_shape=(total_length,1), name='conv4layer'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Lambda(lambda x: apply_phaseshuffle(x, 1)))

    model.add(tf.keras.layers.ZeroPadding1D(padding=1,input_shape=(total_length,1)))
    model.add(tf.keras.layers.Conv1D(1, 3, strides=1, activation=tf.keras.activations.relu, input_shape=(total_length,1), name='conv5layer'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dropout(rate=0.1))
    # model.add(tf.keras.layers.Dense(16, activation="tanh"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # return model

def disc_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def disc_optimizer():
    return tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

def gen_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def gen_optimizer():
    return tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.9, nesterov=True)

#Training loop
def train_step(realAudio, gen, disc, queue, epoch, epochs, VL, beginTraining):
    gen_input = np.array([tf.random.normal([normal_sample,1]) for i in range(gen_num)])
    start = time.time()

    with tf.GradientTape() as gen_grad, tf.GradientTape() as disc_grad:
        generatedAudio = gen(gen_input, training=True)
        # print(generatedAudio.shape)
        real_output = disc(realAudio, training=True)
        fake_output = disc(generatedAudio, training=True)

        gloss = gen_loss(fake_output)
        dloss = disc_loss(real_output, fake_output)

        genWeightChange = gen_grad.gradient(gloss, gen.trainable_variables)
        discWeightChange = disc_grad.gradient(dloss, disc.trainable_variables)

        gen_optimizer().apply_gradients(zip(genWeightChange, gen.trainable_variables))
        disc_optimizer().apply_gradients(zip(discWeightChange, disc.trainable_variables))

        # print(VL, "hfsafjs;adfjsdsal")

        if not VL:
            message = f"Epoch {epoch}/{epochs}:\nGenerator Loss: {gloss} \nDiscriminator Loss: {dloss}\nReal Match (Ideally Not ~100%): {real_output[0][0]*100}% \nFake Match (Ideally = Real Match): {fake_output[0][0]*100}%"
            queue.put(message)
        else:
            end = time.time()
            epochCt = epoch + 1
            message = f"Epoch {epoch}/{epochs}:\nGenerator Loss: {gloss} \nDiscriminator Loss: {dloss}\nReal Match (Ideally Not ~100%): {real_output[0][0]*100}% \nFake Match (Ideally = Real Match): {fake_output[0][0]*100}%\nTime taken this epoch: {end - start} seconds\nAverage time taken per epoch: {(average:=((total:=(end - beginTraining))/epochCt))} seconds\nTotal time taken: {total} seconds\nEstimated time left: {average*(epochs-epochCt)} seconds"
            # print(message)
            queue.put(message)

def train(dataset, epochs, queue, continuelog, outpath, sampling_iter, VL):
    # print(tf.random.normal([1,normal_sample]))
    gc.collect()

    try:
        gen = tf.keras.models.load_model('generator.keras')   
        disc = tf.keras.models.load_model('discriminator.keras')
    except OSError:
        queue.put("Status: Error loading models, try pressing the stop button and start the generation process again.")

    queue.put("Status: Models have been loaded")
    
    while not continuelog.empty():
        continuelog.get(block=False)

    beginTraining = time.time()
    willContinue = True
    for e in range(epochs+1):
        # print(e)
        # print(f"This epoch {e}:")
        if not continuelog.empty():
            willContinue = continuelog.get()
            if not willContinue:
                break
        # print(e)
        train_step(dataset, gen, disc, queue, e, epochs, VL, beginTraining)

        # print(f"Time taken for complete epoch: {time.time() - start} seconds")
        # print(f"Average time taken for 1 train step: {(time.time() - start)/(len(dataset))}\n")
        if e % sampling_iter == 0:
            for i in range(10):
                gen_input = tf.random.normal([1,normal_sample])
                generatedAudio = gen(gen_input)
                output = np.float32(np.float32(generatedAudio[0]))
                rate = 44100
                wavfile.write(f'{outpath}{e}G{i}.wav', rate, output) #-file name- epoch number Generated .wav 
    
    if willContinue:
        for i in range(10):
            gen_input = tf.random.normal([1,normal_sample])
            generatedAudio = gen(gen_input)
            output = np.float32(np.float32(generatedAudio[0]))
            rate = 44100
            wavfile.write(f'{outpath}FinalG{i}.wav', rate, output)
        # print(f"total time taken: {time.time()-begin} seconds")
        # print(f"average time per epoch: {(time.time()-begin)/epochs} seconds")