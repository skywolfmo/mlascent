import tensorflow as tf

class AutoEncoder:
    def __init__(self, input_shape, input_img, encoding_dim, numFilters, dropouts, doBatchNorm): #define Input outside the UNET
        self.input_shape = input_shape
        self.numFilters = numFilters
        self.encoding_dim = encoding_dim
        self.doBatchNorm = doBatchNorm
        self.dropouts = dropouts
        
        self.model = None

#         
        self.build_autoencoder(input_shape, input_img, encoding_dim, numFilters, dropouts, doBatchNorm)

        
    # defining autoencoder model
    def Conv2dBlock(self, inputTensor, numFilters, kernelSize = 3, doBatchNorm = True, name='D'):
        #first Conv
        x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                                  kernel_initializer = 'he_normal', padding = 'same', name=name+'_1') (inputTensor)

        if doBatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)

        x =tf.keras.layers.Activation('relu')(x)

        #Second Conv
        x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                                  kernel_initializer = 'he_normal', padding = 'same', name=name+'_2') (x)
        if doBatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)

        return x


#         self.classifier = tf.keras.Model(inputs = [input_img], outputs = [classifier])
    def build_autoencoder(self, input_shape, input_img, numFilters, droupouts, doBatchNorm):
        # Encoder Part
        c1 = self.Conv2dBlock(input_img, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm, name='encoder_conv2d_1')
        p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
        p1 = tf.keras.layers.Dropout(droupouts)(p1)

        c2 = self.Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm, name='encoder_conv2d_2')
        p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
        p2 = tf.keras.layers.Dropout(droupouts)(p2)

        c3 = self.Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm, name='encoder_conv2d_3')
        p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
        p3 = tf.keras.layers.Dropout(droupouts)(p3)

        c4 = self.Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm, name='encoder_conv2d_4')
        p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
        p4 = tf.keras.layers.Dropout(droupouts)(p4)
        # Latent Space
        c5 = self.Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm, name='encoder_conv2d_5')
        
#       
        # defining Decoder path
        u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = tf.keras.layers.Dropout(droupouts)(u6)
        c6 = self.Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm, name='decoder_conv2d_1')

        u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        c7 = self.Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm, name='decoder_conv2d_2')

        u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = tf.keras.layers.Dropout(droupouts)(u8)
        c8 = self.Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm, name='decoder_conv2d_3')

        u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = tf.keras.layers.Dropout(droupouts)(u9)
        c9 = self.Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm, name='decoder_conv2d_4')
        
        # image reconstructor
        number_of_channels = input_shape[-1]
        image_reconstructor_layer = tf.keras.layers.Conv2D(input_shape[-1], (1, 1), activation='tanh', name='im_rec')(c9)
        self.image_reconstructor_layer = image_reconstructor_layer 
        self.model = tf.keras.Model(inputs = [input_img], outputs = [image_reconstructor_layer])