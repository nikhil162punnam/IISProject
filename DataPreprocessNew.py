#!/usr/bin/env python
# coding: utf-8

# In[59]:
"""
    Assumed directory structure is:
    .
    ├── bosphorusDB
    │   ├── __images__
    │   │   ├── img5.jpg
    │   │   ├── imtab_au_files_small.jpg
    │   │   ├── imtab_emotion_files_small.jpg
    │   │   ├── imtab_headpose_files_small.jpg
    │   │   ├── imtab_mixed1_files_small.jpg
    │   │   └── imtab_occlusion_files_smalll.jpg
    │   └── __others__
    │       ├── bospcrvalsetups
    │       │   ├── bospausDetB.is
    │       │   ├── bospausDetB.mat
    │       │   ├── bospausDetCDE.is
    │       │   ├── bospausDetCDE.mat
    │       │   ├── bospausEstABCDE.is
    │       │   ├── bospausEstABCDE.mat
    │       │   └── readme.txt
    │       ├── bospcrvalsetups.zip
    │       ├── bospdb_agreement.pdf
    │       ├── BosphorusDB
    │       │   └── bs000
    │       │       ├── bs000_E_ANGER_0.lm3
    │       │       ├── bs000_E_DISGUST_0.lm3
    │       │       ├── bs000_E_FEAR_0.lm3
    │       │       ├── bs000_E_HAPPY_0.lm3
    │       │       ├── bs000_E_SADNESS_0.lm3
    │       │       └── bs000_E_SURPRISE_0.lm3
    │       ├── Bosphorus_Files&Content.pdf
    │       ├── facscodes.zip
    │       ├── fileread.zip
    │       └── unallowed_subjects.txt
    └── DataPreprocessNew.py

"""

import os
import pandas as pd
from fnmatch import fnmatch
import numpy as np
from scipy.spatial.distance import cdist


# In[60]:


files_direc = '/bosphorusDB/__others__/BosphorusDB/'
emotion_label = '_E_'
file_ending = '.lm3'

# Get the current working directory
curr_dir = os.getcwd()
folders = os.listdir(curr_dir + files_direc)

# Encoding the emotions
emotions = {'SURPRISE':0,'ANGER':1,'HAPPY':2,'SADNESS':3,'DISGUST':4,'FEAR':5}

# The features we want from the files
features = ['Outer left eyebrow', 'Middle left eyebrow', 'Inner left eyebrow', 'Inner right eyebrow',
            'Middle right eyebrow', 'Outer right eyebrow', 'Outer left eye corner', 'Inner left eye corner',
            'Inner right eye corner', 'Outer right eye corner', 'Left mouth corner', 'Upper lip outer middle',
            'Right mouth corner', 'Lower lip outer middle', 'Chin middle']

# Setting the column names for the final dataframe
# TODO can be improved by making it automatic
final_columns = ['subject_identifier', 'ole_1', 'ole_2', 'ole_3', 'mle_1', 'mle_2', 'mle_3',
                 'ile_1', 'ile_2', 'ile_3', 'ire_1', 'ire_2', 'ire_3', 'mre_1', 'mre_2', 'mre_3',
                 'ore_1', 'ore_2', 'ore_3', 'olec_1', 'olec_2', 'olec_3', 'ilec_1', 'ilec_2', 'ilec_3',
                 'irec_1', 'irec_2', 'irec_3', 'orec_1', 'orec_2', 'orec_3', 'lmc_1', 'lmc_2', 'lmc_3',
                 'ulom_1', 'ulom_2', 'ulom_3', 'rmc_1','rmc_2', 'rmc_3', 'llom_1', 'llom_2', 'llom_3',
                 'cm_1', 'cm_2', 'cm_3', 'emotion']


# In[61]:


# A list to store the features across all the files
final_list = []
# Iterating through each individual subject
for eachsubject in folders:
    files_names = os.listdir(curr_dir + files_direc + eachsubject)
    for each_file in files_names:
        '''
            each_file = bs014_E_SURPRISE_0.lm3
        '''
        each_file_name = each_file + '*' + emotion_label + '*' + file_ending
        # Matches the emotion label with the file name and proceeds only if it exists. In this case our emotion label is _E_
        if fnmatch(each_file,'*' + emotion_label + '*' + file_ending):
            features_list = [each_file]
            # ['bs014', 'SURPRISE_0.lm3']
            subject_id = each_file.split(emotion_label)[0]
            # ['SURPRISE', '0']
            each_emotion = (each_file.split(emotion_label)[1].split(file_ending)[0].split('_')[0])
            each_emotion_id = int(each_file.split(emotion_label)[1].split(file_ending)[0].split('_')[1])
            print(each_file)
            # ~~/bosphorusDB/__others__/BosphorusDB/bs014/bs014_E_SURPRISE_0.lm3'
            file_location = curr_dir + files_direc + subject_id + '/' + each_file
            # Reading the contents of the file
            with open(file_location, 'r') as rfp:
                # Gives each line in the file as an element
                file_lines = rfp.readlines()
                # Iterating for each feature in the file
                for each_feature in features:
                    try:
                        # If the feature values exists in the file, convert the values from string to float and store them
                        feature_values = file_lines[file_lines.index(each_feature+'\n') +1]
                        final_feature_values = [float(eachvalue) for eachvalue in feature_values.split('\n')[0].split(' ')]
                    except ValueError:
                        # IF the particular feature doesn't exist, then add NAN values
                        final_feature_values = [np.nan]*3
                    # Adding the feature values to the row
                    features_list.extend(final_feature_values)
            # Adding the encoded emotion i.e final target to the row list
            features_list.extend([emotions.get(each_emotion)])
            final_list.append(features_list)


# In[62]:


# Creating the DataFrame to store all the values
final_data = (pd.DataFrame(final_list, columns = final_columns))


# In[63]:


# Computing the distances between landmarks
def compute_dist(feat_1, feat_2, given_df):
    feat1_df = pd.concat([given_df[feat_1+'_1'], given_df[feat_1+'_2'], given_df[feat_1+'_3']], axis = 1)
    feat2_df = pd.concat([given_df[feat_2+'_1'], given_df[feat_2+'_2'], given_df[feat_2+'_3']], axis = 1)
    distm = cdist(feat1_df, feat2_df, metric='euclidean')
    column_name = feat_1 + '_' + feat_2
    return pd.Series(np.diagonal(distm), name =column_name)


# In[64]:


# Landmark pairs to compute distance between
distance_list_new = [('ole', 'mle'), ('ole', 'ile'), ('ile', 'mle'), ('ire', 'mre'), ('ore', 'ire'), ('ore', 'mre'),
                 ('mle', 'mre'), ('ole', 'olec'), ('ore', 'orec'), ('ile', 'ilec'), ('ire', 'irec'),
                 ('lmc', 'rmc'), ('lmc', 'llom'), ('lmc', 'olec'), ('lmc', 'cm'),
                 ('rmc', 'llom'), ('rmc', 'orec'), ('rmc', 'cm'),
                 ('ulom', 'llom'), ('cm', 'mle'), ('cm', 'mre')]


# In[65]:


# Droppping null values before computing the product
new_df = final_data.dropna().reset_index(drop = True)


# In[66]:


# Forming a new dataframe with distances as columns
distance_df = pd.concat([compute_dist(eachpair[0],eachpair[1], new_df) for eachpair in distance_list_new], axis =1)


# In[67]:


# Adding emotion & id to the new distance dataframe
distance_df['emotion'] = new_df['emotion']
distance_df['id'] = new_df['subject_identifier']


# In[68]:


# Storing the dataframe in a csv file
distance_df.to_csv('distance_features_15.csv', index=False)


# In[ ]:
