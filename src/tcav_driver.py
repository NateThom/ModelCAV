import sys
import yaml
import os
import math
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from .
from .opensphere.utils import fill_config

# from test_embeddings import main_worker, parse_args
# from utils import fill_config

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

# get arguments and config
args = parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f, yaml.SafeLoader)
config['data'] = fill_config(config['data'])

if args.proj_dirs:
    config['project']['proj_dirs'] = args.proj_dirs

# Step 1: Train opensphere model to perform face verification

# Step 2: Collect samples of various identities from existing datasets
# Read source annotation data
source_annotation_df = pd.read_csv(config["data"]["source_ann_path"])

# Get unique identities
identities = np.unique(np.concatenate([source_annotation_df["INDIVIDUAL_1"], source_annotation_df["INDIVIDUAL_2"]]))
# identities = pd.concat([source_annotation_df["INDIVIDUAL_1"], source_annotation_df["INDIVIDUAL_2"]]).unique()

# Randomly select identities
num_identities_to_select = config["data"]["num_identities_to_select"]
selected_identities = np.random.choice(identities, size=num_identities_to_select, replace=False)

# Initialize the 'test' list in config["data"]
config["data"]["test"] = []

# Generic dataloader dictionary to add to config
generic_dataloader = {
    "type": "DataLoader",
    "batch_size": 4096,
    "shuffle": False,
    "drop_last": False,
    "num_workers": 4
}

# Check if the embeddings have already been extracted for the current data
embeddings_pickle_path = config["data"]["source_name"]+".pkl"
if not os.path.isfile(embeddings_pickle_path):
    # Loop through selected identities and create datasets
    for identity in tqdm(selected_identities):
        # identity = '_'.join(identity.split(" "))
        
        current_id_dataset = {
            "type": "IndividualDataset", 
            "name": '_'.join(identity.split(" ")), 
            "data_dir": "/home/nthom/Documents/opensphere/data/identity/", 
            "ann_path": f"/home/nthom/Documents/opensphere/data/identity/{'_'.join(identity.split(' '))}_ann.txt",
            "test_mode": True,
        }
        config["data"]["test"].append(
            {
                "dataset": current_id_dataset, 
                "dataloader": generic_dataloader
            }
        )

        # Create identity directory if it doesn't exist

        data_dir_path = current_id_dataset["data_dir"] + "_".join(identity.split(" "))
        if not os.path.isdir(data_dir_path):
            os.mkdir(data_dir_path)

            # Get images associated with the current identity
            images_df = pd.concat([
                source_annotation_df[source_annotation_df["INDIVIDUAL_1"] == identity][["INDIVIDUAL_1", "IMAGE_1"]],
                source_annotation_df[source_annotation_df["INDIVIDUAL_2"] == identity][["INDIVIDUAL_2", "IMAGE_2"]]
            ])
            
            id_images = images_df.apply(lambda row: f"{row['INDIVIDUAL_1']}/{row['IMAGE_1']}" if pd.notnull(row['INDIVIDUAL_1']) else f"{row['INDIVIDUAL_2']}/{row['IMAGE_2']}", axis=1).unique()

            # Process and save images
            ann_output_list = []
            for image in id_images:
                input_path = config["data"]["source_data_path"] + image
                output_path = current_id_dataset["data_dir"] + "_".join(image.split(" "))[:-3] + "bmp"
                open_resize_save_image(input_path, output_path, (112,112))
                ann_output_list.append("/".join(output_path.split("/")[-2:]))
            ann_output_df = pd.DataFrame(ann_output_list)
            ann_output_df.to_csv(current_id_dataset["ann_path"], sep=" ", index=None, header=None)

    # Step 3: Process the image samples with the pretrained opensphere model 
    # and extract embeddings

    model_embeddings = main_worker(config)
    with open(embeddings_pickle_path, 'wb') as f:
        pickle.dump(model_embeddings, f)
else:
    with open(embeddings_pickle_path, 'rb') as f:
        model_embeddings = pickle.load(f)

cav_dict = {}

for identity in tqdm(model_embeddings):
    print(identity)
    
    concept_samples = model_embeddings[identity].to("cuda")
    concept_labels = torch.ones(concept_samples.shape[0], dtype=torch.float).to("cuda")
    
    # num_to_sample = math.ceil(concept_samples.shape[0]/config["data"]["num_identities_to_select"])
    random_samples = []
    for id in model_embeddings:
        if id != identity:
            # random_indices = torch.randperm(model_embeddings[id].size(0))[:num_to_sample]
            # random_samples.append(model_embeddings[id][random_indices])
            random_samples.append(model_embeddings[id])
    random_samples = torch.cat(random_samples).to("cuda")
    random_labels = torch.zeros(random_samples.shape[0], dtype=torch.float).to("cuda")

    classifier_inputs = torch.cat([concept_samples, random_samples])
    classifier_labels = torch.cat([concept_labels, random_labels])

    binary_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to("cuda")
    binary_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5).to("cuda")

    classifier = nn.Linear(in_features=concept_samples.shape[1], out_features=1).to("cuda")

    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(classifier.parameters(), lr=1000., momentum=0.9)
    optimizer = optim.SGD(classifier.parameters(), lr=10., momentum=0.9)

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
        if iteration % 1000 == 0:
            print(iteration, acc, f1, loss.item())
    print(f"Iteration: {iteration}, Acc.: {acc}, F1: {f1}")
    # print(classifier.weight[0])

    with torch.no_grad():
        avg_dot = 0
        for index in range(len(concept_samples)):
            avg_dot += (
                torch.dot(
                    concept_samples[index], 
                    classifier.weight[0]
                ) + classifier.bias
            )
        avg_dot /= concept_samples.shape[0]
        print(f"Avg. Concept Dot: {avg_dot}")

        print("####################")
        avg_dot = 0
        for index in range(len(random_samples)):
            # print(input.shape, classifier.weight.shape)
            avg_dot += (
                torch.dot(
                    random_samples[index], 
                    classifier.weight[0]
                ) + classifier.bias
            )
            # print(
            #     torch.dot(
            #         random_samples[index], 
            #         classifier.weight[0]
            #     ) + classifier.bias
            # )
        avg_dot /= random_samples.shape[0]
        print(f"Avg. Random Dot: {avg_dot}")

        print("####################")

    cav_dict[identity] = classifier.to("cpu")

# concept_list = [model_embeddings[id] for id in model_embeddings]

# # Step 4: Train a linear classifier
# # Assuming you have labels for your concept and random samples (1 for concept, 0 for random)
# concept_labels = torch.ones(concept_list[0].shape[0], dtype=torch.float)
# random_labels = torch.zeros(torch.cat(concept_list[1:-1]).shape[0], dtype=torch.float)

# classifier_inputs = torch.cat(concept_list[:-1], dim=0).to("cuda")
# classifier_labels = torch.cat([concept_labels, random_labels], dim=0).to("cuda")
# classifier = nn.Linear(in_features=concept_list[0].shape[1], out_features=1).to("cuda")

# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# # Training loop
# acc_flag = True
# iterations = 0
# while acc_flag and (iterations<1000000):
#     iterations += 1
#     optimizer.zero_grad()
#     logits = torch.squeeze(classifier(classifier_inputs))
#     loss = loss_fn(logits, classifier_labels)
#     loss.backward()
#     optimizer.step()

#     predictions = torch.logical_not(torch.lt(logits, 0)).float()
#     acc = (predictions == classifier_labels).sum()/classifier_inputs.shape[0]
#     if acc >= 1:
#         break
# print(iterations, acc)

# avg_dot = 0
# for index in range(len(concept_list[0])):
#     # print(input.shape, classifier.weight.shape)
#     avg_dot += (
#         torch.dot(
#             classifier_inputs[index], 
#             classifier.weight[0]
#         ) + classifier.bias
#     )
#     print(
#         torch.dot(
#             classifier_inputs[index], 
#             classifier.weight[0]
#         ) + classifier.bias
#     )
# avg_dot /= concept_list[0].shape[0]

# print("####################")
# for index in range(len(concept_list[-1])):
#     # print(input.shape, classifier.weight.shape)
#     print(torch.dot(classifier_inputs[index], classifier.weight[0]) + classifier.bias)

# # Step 5: Find the Concept Activation Vector (CAV)

# # Compute the CAV as the normalized difference between concept and random gradients

# # Step 6: Compute the directional derivative

# # Step 7: Compute the TCAV score
