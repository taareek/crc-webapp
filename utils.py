import feat_extractor
import pandas as pd
import pickle
from skimage import io as imp
import numpy as np
import io
import base64
from PIL import Image

# load feature RF-RFE feature selector 
fs_path = "./models/rfe.pkl"
fs_in = open(fs_path, 'rb')
fs_rfe = pickle.load(fs_in)

## loading base models 
# knn
knn_model_path = "./models/knn.pkl"
knn_in = open(knn_model_path, 'rb')
knn_model = pickle.load(knn_in)

#svm
svm_model_path = "./models/svm.pkl"
svm_in = open(svm_model_path, 'rb')
svm_model = pickle.load(svm_in)

#rfc
rfc_model_path = "./models/rfc.pkl"
rfc_in = open(rfc_model_path, 'rb')
rfc_model = pickle.load(rfc_in)

#mlp
mlp_model_path = "./models/mlp.pkl"
mlp_in = open(mlp_model_path, 'rb')
mlp_model = pickle.load(mlp_in) 


## loading meta model
stack_model_path = "./models/fs_best_40.pkl"
stack_in = open(stack_model_path, 'rb')
stack_model = pickle.load(stack_in)


# defining label dictionary 
label_dict = {
    0: "TUMOR",
    1: "STROMA",
    2: "COMPLEX",
    3: "LYMPHO",
    4: "DEBRIS",
    5: "MUCOSA",
    6: "ADIPOSE",
    7: "EMPTY"
}

# image transformation 
def img_transform(input_img):
    img_bytes = input_img.file.read()
    img = Image.open(io.BytesIO(img_bytes))  # it needs then convert into numpy to work with opencv
    cv_img = np.array(img)

    # encoded image 
    encoded_string = base64.b64encode(img_bytes)
    bs64 = encoded_string.decode('utf-8')
    encoded_img = f'data:image/jpeg;base64,{bs64}'
    return cv_img, encoded_img


# img = imp.imread('./static/image1.png',0)

# getting features from the input image 
# features = feat_extractor.get_features(img).flatten()

# making a dataframe 
# f_data = []
# header = [f"f_{i}" for i in range(1, 1025)]
# f_data.append(features)
# f_df = pd.DataFrame(f_data, columns=header)
# print("original features: ",features.shape)

# feature selection 
def get_selected_ft(all_features):
    f_data = []
    header = [f"f_{i}" for i in range(1, 1025)]
    f_data.append(all_features)
    df = pd.DataFrame(f_data, columns=header)

    s_f = fs_rfe.support_
    selected_features = []

    for feature, selected in zip(df.columns, s_f):
        if selected:
            selected_features.append(feature)
    df = df[selected_features].copy()
    return df

# getting selected features 
# sf = get_selected_ft(features)
# print("selected feature shape: ", sf.shape)


# get prediction from the stacking-ensemble model 
def get_prediction(img):
    
    # getting features from the input image 
    features = feat_extractor.get_features(img).flatten()
    # getting selected features 
    selected_ft = get_selected_ft(features)

    probs_model1_test = knn_model.predict_proba(selected_ft)
    probs_model2_test = svm_model.predict_proba(selected_ft)
    probs_model3_test = rfc_model.predict_proba(selected_ft)
    probs_model4_test = mlp_model.predict_proba(selected_ft)

    meta_features_test = np.concatenate((probs_model1_test, probs_model2_test, probs_model3_test, probs_model4_test), axis=1)

    pred = stack_model.predict(meta_features_test)

    pred_p = stack_model.predict_proba(meta_features_test)
    pred_p = max(pred_p.flatten())
    # print(pred_p*100)
    pred_proba = round(pred_p * 100, 2)
    # print("The predicted clas is", label_dict[int(pred[0])])
    pred_class = label_dict[int(pred[0])]

    return pred_class, pred_proba

# print("Final prediction: \n")
# get_prediction(sf)

## get prediction results by input image
def get_results(input_img, is_api=False):
    # read image 
    org_img, org_encoded_img = img_transform(input_img)
    pred_class, pred_proba = get_prediction(org_img)

    pred_results = {
            "class_name": pred_class,
            "class_probability": pred_proba
        }
    
    # conditionally add image data to the result dictionary
    if not is_api:
        pred_results["org_encoded_img"] = org_encoded_img

    return pred_results