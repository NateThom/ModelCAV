import os
import pandas as pd

class CavPrediction:
    def __init__(self, dataset_embeddings, pred_labels_pairs, cavs_path):
        self.path_cavs = cavs_path
        self.embeddings_dict = dataset_embeddings
        self.pred_labels_pairs = pred_labels_pairs

    def create_feature_vectors(self):
        CAV_vectors = []
        cav_id_img_name = []
        for cav_idx, cav_pickle in enumerate(os.listdir(self.path_cavs)):
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
        # in the format with rows being 1 image, and samples being 
        pd.DataFrame.to_csv(pd.DataFrame(cav_id_img_name), index=False, header=False)
