import subprocess
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import pickle

"""
    Execute the Computer Vision code.
    Change the python environment and code file
"""
def execute():
    popen = subprocess.Popen(['/home/nikhil/Semester_II/Natural_ML/testvenv/bin/python', '/home/nikhil/Semester_II/IIS/Project/Final/IISProject/CVsample.py'],
                             stdout=subprocess.PIPE, universal_newlines=True)

    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
"""
    Column names of the features. Should be given in the same order by CV group!!
"""
columnNames = ['ole_1', 'ole_2', 'ole_3', 'mle_1', 'mle_2', 'mle_3',
'ile_1', 'ile_2', 'ile_3', 'ire_1', 'ire_2', 'ire_3', 'mre_1', 'mre_2', 'mre_3',
'ore_1', 'ore_2', 'ore_3', 'olec_1', 'olec_2', 'olec_3', 'ilec_1', 'ilec_2', 'ilec_3',
'irec_1', 'irec_2', 'irec_3', 'orec_1', 'orec_2', 'orec_3', 'lmc_1', 'lmc_2', 'lmc_3',
'ulom_1', 'ulom_2', 'ulom_3', 'rmc_1','rmc_2', 'rmc_3', 'llom_1', 'llom_2', 'llom_3',
'cm_1', 'cm_2', 'cm_3']

"""
    Distance between the landmarks that we used in our model.
"""
distance_list_new = [('ole', 'mle'), ('ole', 'ile'), ('ile', 'mle'), ('ire', 'mre'), ('ore', 'ire'), ('ore', 'mre'),
                 ('mle', 'mre'), ('ole', 'olec'), ('ore', 'orec'), ('ile', 'ilec'), ('ire', 'irec'),
                 ('lmc', 'rmc'), ('lmc', 'llom'), ('lmc', 'olec'), ('lmc', 'cm'),
                 ('rmc', 'llom'), ('rmc', 'orec'), ('rmc', 'cm'),
                 ('ulom', 'llom'), ('cm', 'mle'), ('cm', 'mre')]
"""
    Emotion legend for our model.
"""
emotions = {0:'SURPRISE',1:'ANGER',2:'HAPPY',3:'SADNESS',4:'DISGUST',5:'FEAR'}

# Loading the machine learning model
with open('finalVotingModel.pkl','rb') as fp:
    model = pickle.load(fp)

# Loading the data scaler
with open('finalScaler.pkl','rb') as fp:
    scaler = pickle.load(fp)

# Compute the euclidean distance between the landmarks
def compute_dist(feat_1, feat_2, given_df):
    feat1_df = pd.concat([given_df[feat_1+'_1'], given_df[feat_1+'_2'], given_df[feat_1+'_3']], axis = 1)
    feat2_df = pd.concat([given_df[feat_2+'_1'], given_df[feat_2+'_2'], given_df[feat_2+'_3']], axis = 1)
    distm = cdist(feat1_df, feat2_df, metric='euclidean')
    column_name = feat_1 + '_' + feat_2
    return pd.Series(np.diagonal(distm), name =column_name)

# Compute the final feature dataframe
def compute_feature(newList):
    final_data = (pd.DataFrame([newList], columns = columnNames))
    distance_df = pd.concat([compute_dist(eachpair[0],eachpair[1], final_data) for eachpair in distance_list_new], axis =1)
    print('The distance features we computed from the co-ordinates are:')
    print(distance_df,'\n')
    return distance_df

# Predict the emotion based on the feature
def predict_emotion(feat_df):
    x_test = scaler.transform(feat_df)
    return emotions.get(model.predict(x_test)[0])

# Main code to execute, waits for the input from CV code.
cds = []
for path in execute():
    if(path.strip() == 'EOF'):
        print('The co-ordinates from computer vision group are :')
        print(cds,'\n')
        feat_df = compute_feature(cds)
        print('The emotion for the given set of landmarks is:',predict_emotion(feat_df),'\n')
        cds = []
    else:
        cds.extend([ float(eachvalue.replace('[','').replace(']','')) for eachvalue in path.strip().split() if(eachvalue.replace('[','').replace(']','')!='')])
