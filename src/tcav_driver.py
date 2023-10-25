import sys
import yaml
import os
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "./src/opensphere/")
# from test_individual_dataset import main_worker, parse_args
import test_individual_dataset
import test_other_dataset
from dataset.utils import get_metrics

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

def open_resize_save_image(input_path, output_path, new_size):
    try:
        # Open the image file
        image = Image.open(input_path)

        # Resize the image to the desired size
        resized_image = image.resize(new_size)

        # Save the resized image to the output path
        resized_image.save(output_path)

        # print("Image successfully resized and saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_datasets(concepts, config):
    if not os.path.isdir(config["CAV_data"]["concept_output_path"]):
        os.mkdir(config["CAV_data"]["concept_output_path"])
    
    # Initialize the 'test' list in config["data"]
    config["data"] = {}
    config["data"]["test"] = []
    # Generic dataloader dictionary to add to config
    generic_dataloader = {
        "type": "DataLoader",
        "batch_size": 4096,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 4
    }
    # Loop through selected identities and create datasets
    for concept in tqdm(concepts):        
        current_concept_dataset = {
            "type": "ConceptDataset", 
            "name": '_'.join(concept.split(" ")), 
            "data_dir": config['CAV_data']['concept_output_path'], 
            "ann_path": f"{config['CAV_data']['concept_output_path']}{'_'.join(concept.split(' '))}_ann.txt",
            "test_mode": True,
        }
        config["data"]["test"].append(
            {
                "dataset": current_concept_dataset, 
                "dataloader": generic_dataloader
            }
        )

        # Create identity directory if it doesn't exist
        data_dir_path = current_concept_dataset["data_dir"] + "_".join(concept.split(" "))
        if not os.path.isdir(data_dir_path):
            os.mkdir(data_dir_path)

            id_images = os.listdir(config["CAV_data"]["concept_data_path"] + concept)
            id_images = [f"{concept}/{img}" for img in id_images]

            # Process and save images
            ann_output_list = []
            for image in id_images:
                input_path = config["CAV_data"]["concept_data_path"] + image
                output_path = current_concept_dataset["data_dir"] + "_".join(image.split(" "))[:-3] + "bmp"
                open_resize_save_image(input_path, output_path, (112,112))
                ann_output_list.append("/".join(output_path.split("/")[-2:]))
            ann_output_df = pd.DataFrame(ann_output_list)
            ann_output_df.to_csv(current_concept_dataset["ann_path"], sep=" ", index=None, header=None)
    return config

def create_CAVs(model_embeddings, config):    
    for concept in tqdm(model_embeddings):
        if os.path.isfile(f"CAVs/{config['CAV_data']['source_name']}/{concept}.pth"):
            continue

        concept_samples = model_embeddings[concept].to("cuda")
        concept_labels = torch.ones(concept_samples.shape[0], dtype=torch.float).to("cuda")

        random_samples = []
        for i in model_embeddings:
            if i != concept:
                random_samples.append(model_embeddings[i])
        random_samples = torch.cat(random_samples).to("cuda")
        random_labels = torch.zeros(random_samples.shape[0], dtype=torch.float).to("cuda")

        classifier_inputs = torch.cat([concept_samples, random_samples])
        classifier_labels = torch.cat([concept_labels, random_labels])

        binary_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to("cuda")
        binary_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5).to("cuda")

        # classifier = nn.Linear(in_features=concept_samples.shape[1], out_features=1).to("cuda")
        classifier = LinearClassifier(concept_samples.shape[1], 1).to("cuda")

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(classifier.parameters(), lr=1000., momentum=0.9)
        # optimizer = optim.SGD(classifier.parameters(), lr=10., momentum=0.9)

        # Training loop
        acc_flag = True
        iteration = 0
        while acc_flag and (iteration<1000000):
            iteration += 1
            optimizer.zero_grad()
            logits = torch.squeeze(classifier(classifier_inputs))
            loss = loss_fn(logits, classifier_labels)
            loss.backward()
            optimizer.step()

            predictions = torch.logical_not(torch.lt(logits, 0.5)).float()
            acc = binary_acc(predictions, classifier_labels)
            f1 = binary_f1(predictions, classifier_labels)
            if acc >= 1:
                break
            # if iteration % 1000 == 0:
            #     print(iteration, acc, f1, loss.item())
        # print(f"Iteration: {iteration}, Acc.: {acc}, F1: {f1}")
        torch.save(
            classifier.state_dict(), 
            f"CAVs/{config['CAV_data']['source_name']}/{concept}.pth"
        )

def load_CAVs(selected_concepts, config):
    cav_dict = {}
    for concept in selected_concepts:
        CAV = LinearClassifier(512, 1)
        CAV.load_state_dict(
            torch.load(f"CAVs/{config['CAV_data']['source_name']}/{concept}.pth")
        )
        CAV = CAV.to("cuda")
        CAV.eval()
        cav_dict[concept] = CAV
    return cav_dict

def main():
    # get arguments and config
    args = test_individual_dataset.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    if args.proj_dirs:
        config['project']['proj_dirs'] = args.proj_dirs
        
    if not os.path.isdir("./embeddings"):
        os.mkdir("embeddings")
    if not os.path.isdir("./CAVs"):
        os.mkdir("CAVs")
    if not os.path.isdir(f"./CAVs/{config['CAV_data']['source_name']}"):
        os.mkdir(f"./CAVs/{config['CAV_data']['source_name']}")

    concepts = os.listdir(config["CAV_data"]["concept_data_path"])

    embeddings_pickle_path = "./embeddings/" + config["CAV_data"]["source_name"] + ".pkl"
    if not os.path.isfile(embeddings_pickle_path):
        config = create_datasets(concepts, config)

        model_embeddings = test_individual_dataset.main_worker(config)
        with open(embeddings_pickle_path, 'wb') as f:
            pickle.dump(model_embeddings, f)
    else:
        with open(embeddings_pickle_path, 'rb') as f:
            model_embeddings = pickle.load(f)

    create_CAVs(model_embeddings, config)

    selected_concepts = random.sample(
        list(model_embeddings), 
        config["CAV_data"]["num_concepts_to_select"]
    )
    cav_dict = load_CAVs(selected_concepts, config)
    # print(cav_dict)

    config["data"] = {}
    config["data"]["test"] = config["model_predictions_data"]

    model_embedding_pairs, model_predictions, model_labels = test_other_dataset.main_worker(config)
    
    results = get_metrics(
        model_labels["LFW"], 
        model_predictions["LFW"], 
        ['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']
    )
    
    temp=1

if __name__=="__main__":
    main()
