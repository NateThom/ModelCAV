import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn import svm


# calculate cav feature vectors for all images, and create abs sample feature vectors
class CavPrediction:
    def __init__(self, dataset_embeddings, pred_labels_pairs, cavs_path):
        self.path_cavs = cavs_path
        self.cav_names = os.listdir(self.path_cavs)
        self.embeddings_dict = dataset_embeddings
        self.pred_labels_pairs = pred_labels_pairs

    # make initial feature vectors for all images using cavs and embeddings
    def create_feature_vectors(self, path_to_feature_save):
        self.cav_feature_save_location = path_to_feature_save
        CAV_vectors = []
        cav_id_img_name = []
        for cav_idx, cav_pickle in enumerate(self.cav_names):
            cav_features = []
            # TODO: open 1 cav
            cav_model = open(cav_pickle)
            for identity_name in self.embeddings_dict:
                for embedding in identity_name:
                    image_name = ""
                    # TODO: pass embeding through cav model, this should return 1 number
                    feature = cav_model * embedding
                    cav_features.append(feature)

                    # only want each image to have 1 row
                    if cav_idx == 0:
                        cav_id_img_name.append([identity_name, image_name])

            CAV_vectors.append(cav_features)
        for cav_id in range(len(CAV_vectors)):
            for cav_feature in range(len(CAV_vectors[cav_id])):
                cav_id_img_name[cav_feature].append(CAV_vectors[cav_id][cav_feature])
        self.cav_columns = [f"CAV_{p}" for p in self.cav_names]
        columns = ["IDENTITIY_NAME", "IMAGE_NAME"] + self.cav_columns
        # in the format with rows being 1 image, and samples being 
        pd.DataFrame(cav_id_img_name, columns=columns).to_csv(self.cav_feature_save_location)

    # make abs diff feature vectors for all samples using cav_feature_vectors and protocol
    def create_abs_feature_vectors(self, path_to_protocol, path_to_abs_feature_vectors):
        cav_feat_vectors = self.load_feature_vectors(self.cav_feature_save_location)
        protocol = pd.read_csv(path_to_protocol, header=False).values.tolist()
        abs_feat_vectors = []
        for sample_idx, sample in enumerate(protocol):
            feature_vector_1 = cav_feat_vectors.loc[(cav_feat_vectors["IDENTITIY_NAME"] == sample[0]) & (cav_feat_vectors["IMAGE_NAME"] == sample[1])][self.cav_columns].values.tolist()
            feature_vector_2 = cav_feat_vectors.loc[(cav_feat_vectors["IDENTITIY_NAME"] == sample[2]) & (cav_feat_vectors["IMAGE_NAME"] == sample[3])][self.cav_columns].values.tolist()
            if len(feature_vector_1) == len(self.cav_columns) and len(feature_vector_2) == len(self.cav_columns):
                feature_vector_abs = [abs(feature_vector_1[idx] - feature_vector_2[idx]) for idx in range(len(feature_vector_1))]
                abs_feat_vectors.append(sample + feature_vector_abs)
            else:
                print("ERROR: bad feature lengths for ", sample)
                return None
        columns = ["IDENTITIY_NAME_1", "IMAGE_NAME_1", "IDENTITIY_NAME_2", "IMAGE_NAME_2", "LABEL", "SPLIT"] + self.cav_columns
        pd.DataFrame(abs_feat_vectors, columns=columns).to_csv(self.cav_feature_save_location)


    def load_feature_vectors(self, path_to_feature_vectors):
        return pd.read_csv(path_to_feature_vectors)
    

class ModelPredictorWithCAVS:
    def __init__(self, path_to_abs_feature_vectors, path_to_protocol, test_split):
        self.path_to_abs_feature_vectors = path_to_abs_feature_vectors
        self.path_to_protocol = path_to_protocol
        self.test_split = test_split
        self.train_dataset = self.ABS_Feature_Vector_Dataset(self.path_to_abs_feature_vectors, self.path_to_protocol, self.test_split, testing=False)
        self.test_dataset = self.ABS_Feature_Vector_Dataset(self.path_to_abs_feature_vectors, self.path_to_protocol, self.test_split, testing=True)
        self.model = svm.SVC(C=100, gamma=.001)

    class ABS_Feature_Vector_Dataset(Dataset):
        def __init__(self, path_to_abs_feature_vectors, path_to_protocol, test_split, testing):
            self.abs_feature_vectors = pd.read_csv(path_to_abs_feature_vectors)
            self.protocol = pd.read_csv(path_to_protocol, header=False).values.tolist()
            self.test_split = test_split
            self.samples = self.abs_feature_vectors[self.abs_feature_vectors.split == test_split if testing else self.abs_feature_vectors.split != test_split].values.tolist()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, id):
            feature_vector = self.samples[id]
            label = feature_vector[5]
            numerical_vector = feature_vector[6:]
            return torch.as_tensor(numerical_vector), torch.as_tensor(label)
        
        def getAll(self):
            labels = [data[5] for data in self.samples]
            feature_vectors = [data[6:] for data in self.samples]
            return feature_vectors, labels

    def train(self):
        self.model.fit(self.train_dataset.getAll())

    def test(self):
        test_data, test_labels = self.test_dataset.getAll()
        print(self.model.predict(test_data[0]))