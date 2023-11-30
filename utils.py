import feat_extractor
import pandas as pd
import pickle
from skimage import io
import numpy as np

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


img = io.imread('./static/image1.png',0)

# getting features from the input image 
features = feat_extractor.get_features(img).flatten()

# making a dataframe 
f_data = []
header = [f"f_{i}" for i in range(1, 1025)]
f_data.append(features)
f_df = pd.DataFrame(f_data, columns=header)
print("original features: ",features.shape)

# feature selection 
def get_selected_ft(df):
    s_f = fs_rfe.support_
    selected_features = []

    for feature, selected in zip(df.columns, s_f):
        if selected:
            selected_features.append(feature)
    df = df[selected_features].copy()
    return df

# getting selected features 
sf = get_selected_ft(f_df)
print("selected feature shape: ", sf.shape)


# get prediction from the stacking-ensemble model 
def get_prediction(selected_ft):
    probs_model1_test = knn_model.predict_proba(selected_ft)
    probs_model2_test = svm_model.predict_proba(selected_ft)
    probs_model3_test = rfc_model.predict_proba(selected_ft)
    probs_model4_test = mlp_model.predict_proba(selected_ft)

    meta_features_test = np.concatenate((probs_model1_test, probs_model2_test, probs_model3_test, probs_model4_test), axis=1)

    pred = stack_model.predict(meta_features_test)

    pred_p = stack_model.predict_proba(meta_features_test)
    pred_p = max(pred_p.flatten())
    print(pred_p*100)
    print("The predicted clas is", label_dict[int(pred[0])])

print("Final prediction: \n")
get_prediction(sf)