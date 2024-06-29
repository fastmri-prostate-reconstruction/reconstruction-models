import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from functools import partial

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
# from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
# import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Lambda, concatenate, Add, Conv2D

def to_complex(x, n):
    return tf.complex(
        tf.cast(x[..., :n], dtype=tf.float32),
        tf.cast(x[..., n:], dtype=tf.float32),
    )

def to_real(x):
    return tf.concat([
        tf.math.real(x),
        tf.math.imag(x),
    ], axis=-1)

def _concatenate_real_imag(x):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    return concatenate([x_real, x_imag])

def _complex_from_half(x, n, output_shape):
    return Lambda(lambda x: to_complex(x, n), output_shape=output_shape)(x)



def conv2d_complex(x, n_filters, n_convs, activation='relu', output_shape=None, res=False, last_kernel_size=3):
    x_real_imag = _concatenate_real_imag(x)
    n_complex = output_shape[-1]
    for j in range(n_convs):
        x_real_imag = Conv2D(
            n_filters,
            3,
            activation=activation,
            padding='same',
            kernel_initializer='glorot_uniform',
            # kernel_regularizer=regularizers.l2(1e-6),
            # bias_regularizer=regularizers.l2(1e-6),
        )(x_real_imag)
    x_real_imag = Conv2D(
        2 * n_complex,
        last_kernel_size,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
        # kernel_regularizer=regularizers.l2(1e-6),
        # bias_regularizer=regularizers.l2(1e-6),
    )(x_real_imag)
    x_real_imag = _complex_from_half(x_real_imag, n_complex, output_shape)
    if res:
        x_final = Add()([x, x_real_imag])
    else:
        x_final = x_real_imag
    return x_final


from tensorflow.keras.layers import Layer, Lambda, Add

class MultiplyScalar(Layer):
    def __init__(self, **kwargs):
        super(MultiplyScalar, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.sample_weight = self.add_weight(
            name='sample_weight',
            shape=(1,),
            initializer='ones',
            trainable=True,
        )
        super(MultiplyScalar, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.cast(self.sample_weight, tf.complex64) * x

    def compute_output_shape(self, input_shape):
        return input_shape

def _replace_values_on_mask(x):
    # TODO: check in multicoil case
    cnn_fft, kspace_input, mask = x
    anti_mask = tf.expand_dims(tf.dtypes.cast(1.0 - mask, cnn_fft.dtype), axis=-1)
    replace_cnn_fft = tf.math.multiply(anti_mask, cnn_fft) + kspace_input
    return replace_cnn_fft

def enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar=None, noiseless=True):
    if noiseless:
        data_consistent_kspace = Lambda(_replace_values_on_mask, output_shape=input_size)([kspace, kspace_input, mask])
    else:
        if multiply_scalar is None:
            multiply_scalar = MultiplyScalar()
        kspace_masked = Lambda(lambda x: -_mask_tf(x), output_shape=input_size)([kspace, mask])
        data_consistent_kspace = Add()([kspace_input, kspace_masked])
        data_consistent_kspace = multiply_scalar(data_consistent_kspace)
        data_consistent_kspace = Add()([data_consistent_kspace, kspace])
    return data_consistent_kspace

def default_model_compile(model, lr, loss='mean_absolute_error'):
    opt_kwargs = {}
    precision_policy = mixed_precision.global_policy()

    if loss == 'compound_mssim':
        loss = compound_l1_mssim_loss
    elif loss == 'mssim':
        loss = partial(compound_l1_mssim_loss, alpha=0.9999)
        loss.__name__ = "mssim"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=loss,
        metrics=['mean_squared_error'],
    )

def _tf_crop(im, cropx=320, cropy=None):
    if cropy is None:
        cropy = cropx
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    starty = tf.cond(
        tf.equal(tf.math.mod(cropy, 2), 0),
        lambda: starty,
        lambda: starty -1,
    )
    im = im[:, starty:starty+cropy, startx:startx+cropx, :]
    return im

def tf_fastmri_format(image):
    image = Lambda(lambda x: _tf_crop(tf.math.abs(x)), name='cropping', output_shape=(320, 320, 1))(image)
    return image

from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift

def tf_adj_op(y, idx=0):
    x, mask = y
    x_masked = _mask_tf((x, mask))
    x_inv = tf_unmasked_adj_op(x_masked, idx=idx)
    return x_inv


def _mask_tf(x):
    k_data, mask = x
    mask = tf.expand_dims(tf.dtypes.cast(mask, k_data.dtype), axis=-1)
    masked_k_data = tf.math.multiply(mask, k_data)
    return masked_k_data

def tf_unmasked_adj_op(x, idx=0):
    axes = [len(x.shape) - 3, len(x.shape) - 2]
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(x)[-3:-1]), 'float32')), x.dtype)
    return scaling_norm * tf.expand_dims(fftshift(ifft2d(ifftshift(x[..., idx], axes=axes)), axes=axes), axis=-1)


def cascade_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, noiseless=True, lr=1e-3, fastmri=True, activation='relu'):
    r"""This net cascades several convolution blocks followed by data consistency layers

    The original network is described in [S2017]. Its implementation is
    available at https://github.com/js3611/Deep-MRI-Reconstruction in pytorch.

    Parameters:
    input_size (tuple): the size of your input kspace, default to (640, None, 1)
    n_cascade (int): number of cascades (n_c in paper), defaults to 5
    n_convs (int): number of convolution in convolution blocks (n_d + 1 in paper), defaults to 5
    n_filters (int): number of filters in a convolution, defaults to 16
    noiseless (bool): whether the data consistency has to be done in a noiseless
        manner. If noiseless is `False`, the noise level is learned (i.e. lambda
        in paper, is learned). Defaults to `True`.
    lr (float): learning rate, defaults to 1e-3
    fastmri (bool): whether to put the final image in fastMRI format, defaults
        to True (i.e. image will be cropped to 320, 320)
    activation (str or function): see https://keras.io/activations/ for info

    Returns:
    keras.models.Model: the deep cascade net model, compiled
    """
    # inputs
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')

    zero_filled = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple')(kspace_input)

    image = zero_filled
    multiply_scalar = MultiplyScalar()
    for i in range(n_cascade):
        # residual convolution
        image = conv2d_complex(image, n_filters, n_convs, output_shape=input_size, res=True, activation=activation)
        # data consistency layer
        kspace = Lambda(tf_unmasked_op, output_shape=input_size, name='fft_simple_{i}'.format(i=i+1))(image)
        kspace = enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar, noiseless)
        image = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple_{i}'.format(i=i+1))(kspace)
    # module and crop of image
    if fastmri:
        image = tf_fastmri_format(image)
    else:
        image = Lambda(tf.math.abs)(image)
    model = Model(inputs=[kspace_input, mask], outputs=image)

    default_model_compile(model, lr)

    return model

if __name__ == '__main__':
    run_params = {
        'n_cascade': 5,
        'n_convs': 5,
        'n_filters': 48,
        'noiseless': True,
    }

    model = cascade_net()
    model.load_weights('model_weights.h5')


    
