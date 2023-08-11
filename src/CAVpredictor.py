import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score


# calculate cav feature vectors for all images, and create abs sample feature vectors
class CAVFeatureVectorGenerator:
    def __init__(self, dataset_embeddings, pred_labels_pairs, cav_dictionary):
        self.cav_dictionary = cav_dictionary
        self.embeddings = dataset_embeddings
        self.pred_labels_pairs = pred_labels_pairs

    # make initial feature vectors for all images using cavs and embeddings
    def create_feature_vectors(self, path_to_feature_save):
        self.cav_feature_save_location = path_to_feature_save
        CAV_vectors = []
        cav_id_img_name = []
        for cav_idx, cav_name in enumerate(self.cav_dictionary):
            cav_features = []
            cav_model = self.cav_dictionary[cav_name]
            for embedding_idx, embedding in enumerate(self.embeddings):
                # TODO: pass embeding through cav model, this should return 1 number
                feature = cav_model.weights * embedding + cav_model.bias
                cav_features.append(feature)

                # only want each image to have 1 row
                if cav_idx == 0:
                    cav_id_img_name.append([embedding_idx])

            CAV_vectors.append(cav_features)

        for cav_id in range(len(CAV_vectors)):
            for cav_feature in range(len(CAV_vectors[cav_id])):
                cav_id_img_name[cav_feature].append(CAV_vectors[cav_id][cav_feature])

        self.cav_columns = [f"CAV_{p}" for p in self.cav_names]
        columns = ["EMBEDDING_INDEX"] + self.cav_columns
        # in the format with rows being 1 image, and samples being 
        pd.DataFrame(cav_id_img_name, columns=columns).to_csv(self.cav_feature_save_location)

    # make abs diff feature vectors for all samples using cav_feature_vectors and protocol
    def create_abs_feature_vectors(self, path_to_abs_feature_vectors):
        cav_feat_vectors = self.load_feature_vectors(self.cav_feature_save_location)
        abs_feat_vectors = []
        for sample_idx, sample in enumerate(self.pred_labels_pairs):
            embedding_index_1 = sample[1]
            embedding_index_2 = sample[2]
            feature_vector_1 = cav_feat_vectors.loc[(cav_feat_vectors["EMBEDDING_INDEX"] == embedding_index_1)][self.cav_columns].values.tolist()
            feature_vector_2 = cav_feat_vectors.loc[(cav_feat_vectors["EMBEDDING_INDEX"] == embedding_index_2)][self.cav_columns].values.tolist()
            if len(feature_vector_1) == len(self.cav_columns) and len(feature_vector_2) == len(self.cav_columns):
                feature_vector_abs = [abs(feature_vector_1[idx] - feature_vector_2[idx]) for idx in range(len(self.cav_columns))]
                abs_feat_vectors.append(sample + feature_vector_abs)
            else:
                print("ERROR: bad feature lengths for ", sample)
                return None
        columns = ["LABEL", "EMBEDDING_INDEX_1", "EMBEDDING_INDEX_2", "SPLIT"] + self.cav_columns
        pd.DataFrame(abs_feat_vectors, columns=columns).to_csv(path_to_abs_feature_vectors)

    def load_feature_vectors(self, path_to_feature_vectors):
        return pd.read_csv(path_to_feature_vectors)
    

class ModelPredictorWithCAVS:
    def __init__(self, path_to_abs_feature_vectors, test_split):
        self.path_to_abs_feature_vectors = path_to_abs_feature_vectors
        self.test_split = test_split
        self.train_dataset = self.ABS_Feature_Vector_Dataset(self.path_to_abs_feature_vectors, self.test_split, testing=False)
        self.test_dataset = self.ABS_Feature_Vector_Dataset(self.path_to_abs_feature_vectors, self.test_split, testing=True)
        self.model = svm.SVC(C=100, gamma=.001)

    class ABS_Feature_Vector_Dataset(Dataset):
        def __init__(self, path_to_abs_feature_vectors, test_split, testing):
            self.abs_feature_vectors = pd.read_csv(path_to_abs_feature_vectors)
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
        train_data, train_labels = self.train_dataset.getAll()
        self.model.fit(train_data, train_labels)

    def test(self):
        test_data, test_labels = self.test_dataset.getAll()
        predicitons = [self.model.predict(test_feat_vect) for test_feat_vect in test_data]
        return predicitons, test_labels

def main():
    # path to trained cav models
    path_to_cavs = ""
    # path to feature vectors for each image in the dataset, generated using embeddings and cavs
    path_to_cav_feature_vectors = ""
    # path to feature vector for each sample in the given protocol, generated with abs difference of pairs of img_cav_feat_vects
    path_to_abs_feature_vectors = ""

    # TODO: the code after this assumes dataset embeddings, and pred labels pairs are in the desired format, this will need more code to get to this point
    # dictionary of embeddings {identity:{img:[embedding]}}
    dataset_embeddings = {}
    # list of predictions and pair of images [[predicted label, pair 1, pair 2]]
    pred_labels_pairs = []

    # create instance of CAVFeatureVectorGenerator
    cav_feat_vect_generator = CAVFeatureVectorGenerator(dataset_embeddings, pred_labels_pairs, path_to_cavs)
    # generate a feature vector for all images
    cav_feat_vect_generator.create_feature_vectors(path_to_cav_feature_vectors)
    # using generated feature vector for all images, create abs feature vector for each sample in a protocol
    cav_feat_vect_generator.create_abs_feature_vectors(path_to_abs_feature_vectors)

    predictions = []
    pred_labels = []
    for test_split in range(10):
        cav_pred = ModelPredictorWithCAVS(path_to_abs_feature_vectors, test_split)
        cav_pred.train()
        preds, labels = cav_pred.test()
        predictions += preds
        pred_labels += labels

    print("predictions: ", predictions)
    print("labels: ", pred_labels)
    print("lengths: ", len(predictions), len(pred_labels))
    if len(predictions) == len(pred_labels):
        acc = accuracy_score(pred_labels, predictions)
        print("Accuracy: ", acc)

if __name__ == '__main__':
    main()