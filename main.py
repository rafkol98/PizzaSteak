# source ~/.streamlit_ve/bin/activate

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

header = st.beta_container()
body = st.beta_container()


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(img, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """

  # Decode the read file into a tensor & ensure 3 colour channels 
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

with header:
    st.title('🍕Pizza or 🥩Steak')
    st.write('This is an application I quickly developed to test and deploy a Convolutional Neural Network model I created which classifies images between my two different foods, Pizza and Steak!')
with body:
    display = st.checkbox('CNN model code')
    if display:
      st.write("The final deep learning model which drives this web application can be found here: https://colab.research.google.com/drive/1a-zdcBqSWqbm6PsiDoYC_8MyXTC6vmhS?usp=sharing. The final model was developed after extensive experimentation.")

    loaded_model = tf.keras.models.load_model("pizza_steak.h5")
    uploaded_file = st.file_uploader(label="Choose a file", type=['png','jpg','jpeg'])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")

        resized = cv2.resize(opencv_image, (224,224))/255.

        # pizza = load_and_prep_image(resized)
        pred = loaded_model.predict(tf.expand_dims(resized, axis=0))

        label = "Mamma mia! that looks like a delicious Pizza 🍕" if pred <= 0.5 else "That looks like a juicy Steak 🥩, maybe try it with Jack Daniels sauce?" 

        st.subheader(label)
