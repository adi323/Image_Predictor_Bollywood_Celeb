import os
import cv2
import random
from mtcnn import MTCNN
import numpy as np
import pickle
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from keras_vggface.vggface import VGGFace
import streamlit as st

emojilist  = ['😀', '😁', '🤣', '😃', '😄', '😅', '😆', '😉', '😊', '😋', '😎', '😍', '😘', '😗']
detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('features.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def extract_features(x,y,width,height,img_path,model):
    img = cv2.imread(img_path)
    original = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    return original[y:y + height, x:x + width],result


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False



def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    
    if save_uploaded_image(uploaded_image):
        results = detector.detect_faces(cv2.imread(os.path.join('uploads',uploaded_image.name)))
        for i in results:

            x, y, width, height = i['box']
                
            face,features = extract_features(x,y,width,height,os.path.join('uploads',uploaded_image.name),model)
        
            index_pos = recommend(feature_list,features)
            predicted_actor = filenames[index_pos].split('/')[1].replace('_',' ')
        
            col1,col2 = st.columns(2)
            
            lst=random.sample(emojilist,2)
            with col1:
                st.header('We found this {}{}'.format(lst[0],lst[1]))
                st.image(face,use_column_width=True,width=500)
            with col2:
                left,right=st.columns(2)
                with left:
                    st.subheader("You look like")
                with right:
                    st.header(predicted_actor)
                st.image(filenames[index_pos],use_column_width=True,width=500)