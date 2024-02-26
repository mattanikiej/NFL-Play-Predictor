
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

"""
Run file with "python3 kmeans_prediction.py"
"""

def kmeans(train, infer, baseline, k):
    """
    Performs kmeans on input data.
    Input: (list) train, training dataset and labels 
           (list) infer, dataset to be inferred and labels
           (bool) baseline, whether to generate baseline values or not
           (int) k, number of centroidiss
    Output: (list of int) output predictions
            (list of int) output labels
            (float) output percentage of correct predictions 

    Note: input data is in format [[data], label]
    """
    # If generating baseline, create random predictions
    if baseline:
        print("Generating baseline kmeans results...")
        preds = []
        labels = []
        num_correct = 0
        num_total = len(infer)
        for point in infer:
            label = point[1]
            pred = random.choice([0,1])
            preds.append(pred)
            labels.append(int(label))
            if pred == int(label):
                num_correct += 1

        return preds, labels, num_correct/num_total

    # If not generating baseline
    print("Generating kmeans results...")
    # Assign random centroids to each cluster
    centroids = [centroid[0] for centroid in random.sample(train, k)]

    # Train until convergence
    not_converged = True
    epoch = 0
    while not_converged:
        print("Epoch " + str(epoch))

        # Create dicts of centroid indexes and clusters
        centroid_ids = {}
        clusters = {}
        curr_index = 0
        for centroid in centroids:
            centroid_ids[curr_index] = centroid
            clusters[curr_index] = []
            curr_index += 1

        # Assigning points to cluster with closest centroid
        for point in train:
            if point in centroids:
                continue
            else:
                curr_centroid = None
                curr_distance = float("inf")

                # Calculate closest centroid to point
                for centroid in centroids:
                    if dist(point, centroid) < curr_distance:
                        curr_centroid = centroid
                        curr_distance = dist(point, centroid)
                
                # Find which cluster to add point to
                for id in centroid_ids.keys():
                    if centroid_ids[id] == curr_centroid:
                        clusters[id].append(point)

        # Calculate new cluster centroid by averaging over each column
        new_centroids = []
        for i in range(k):
            curr_centroid = centroids[i]
            curr_cluster = clusters[i]
            curr_new_centroid = []
            for i in range(len(curr_centroid)):
                curr_sum = sum([point[0][i] for point in curr_cluster])
                curr_new_centroid.append(curr_sum/len(curr_cluster))
            new_centroids.append(curr_new_centroid)
            
        # Assign new centroids and check for convergence
        if centroids != new_centroids:
            centroids = new_centroids
            epoch += 1
        else:
            not_converged = False
    
    # Check majority label to assign to clusters
    labels_key = []
    for cluster_id in clusters.keys():
        curr_cluster = clusters[cluster_id]
        curr_label_count = [0,0]
        for point in curr_cluster:
            curr_label = point[1]
            curr_label_count[int(curr_label)] += 1
        labels_key.append(curr_label_count.index(max(curr_label_count)))

    
    # Calculate predicted label for each inference sample
    preds = []
    labels = []
    num_correct = 0
    num_total = len(infer)
    for point in infer:
        label = point[1]
        curr_closest_index = None
        curr_distance = float("inf")

        # Calculate closest centroid to point
        for i in range(len(centroids)):
            curr_centroid = centroids[i]
            if dist(point, curr_centroid) < curr_distance:
                curr_closest_index = i
                curr_distance = dist(point, curr_centroid)

        pred = labels_key[curr_closest_index]
        preds.append(pred)
        labels.append(int(label))
        if pred == int(label):
            num_correct += 1

    return preds, labels, num_correct/num_total

# TODO: Add in cosim distance also
def dist(point, centroid):
    """
    Calculates Euclidean distance between a point and a centroid.
    Input: (list of float) point
           (list of float) centroid
    Output: (float) output distance
    """
    sum_of_squares = 0
    for i in range(len(point[0])): # Add -1 because last val is the label
        curr1 = point[0][i]
        curr2 = centroid[i]
        sum_of_squares += pow(curr1 - curr2, 2)
    return pow(sum_of_squares, 0.5)

def print_metrics(output):
    """
    Print evaluation metrics for output of kmeans.
    Input: (list of int) output predictions
           (list of int) output labels
           (float) output percentage of correct predictions 
    output: none, prints precision, recall, F1-scores, confusion matrix, perplexity
    """
    # Read in variables
    preds = output[0]
    labels = output[1]
    correct_rate = output[2]

    # Calculate metrics
    print("Accuracy: " + str(correct_rate))
    print("Precision: " + str(precision_score(labels, preds)))
    print("Recall: " + str(recall_score(labels, preds)))
    print("F1 Score: " + str(f1_score(labels, preds)))
    print("Confusion Matrix: \n" + str(confusion_matrix(labels, preds)))
    print()

if __name__ == "__main__":

    # For reference
    key_dict = {'play_id': 0, 'game_id': 1, 'home_team': 2, 'away_team': 3, 'posteam': 4,
                'posteam_type': 5, 'defteam': 6, 'side_of_field': 7, 'yardline_100': 8,
                'game_date': 9, 'quarter_seconds_remaining': 10, 'half_seconds_remaining': 11,
                'game_seconds_remaining': 12, 'game_half': 13, 'drive': 14, 'qtr': 15, 'down': 16,
                'goal_to_go': 17, 'time': 18, 'yrdln': 19, 'ydstogo': 20, 'ydsnet': 21, 'play_type': 22,
                'no_huddle': 23, 'home_timeouts_remaining': 24, 'away_timeouts_remaining': 25,
                'posteam_timeouts_remaining': 26, 'defteam_timeouts_remaining': 27, 'total_home_score': 28,
                'total_away_score': 29, 'posteam_score': 30, 'defteam_score': 31, 'score_differential': 32}
    
    # Run various models using different features
    col_names_to_use = ["yardline_100", "quarter_seconds_remaining", "game_seconds_remaining", "ydstogo", "score_differential"]
    col_ids_to_use = [[8, 10, 12, 20, 32]]
    
    for col_ids in col_ids_to_use:
        
        # Run kmeans on full dataset
        if col_ids == ["all"]:
            with open("/Users/andrewshen/Desktop/msai349/FinalProject/data/scaled_features.csv", "r") as file:
                dataset = file.readlines()[1:]
                dataset = [[float(val.strip()) for val in line.split(",")] for line in dataset]
            with open("/Users/andrewshen/Desktop/msai349/FinalProject/data/labels.csv", "r") as file:
                labels = file.readlines()[1:]
                labels = [float(label.strip()) for label in labels]

        # Run kmeans on subset of dataset
        else:
            with open("/Users/andrewshen/Desktop/msai349/FinalProject/data/pass_run_pbp_2019_norm.csv", "r") as file:
                dataset = [line.split(",") for line in file.readlines()][1:]
                labels = [line[key_dict['play_type']].strip() for line in dataset]
            dataset_trimmed = []
            for line in dataset:
                curr_line = []
                for i in range(len(line)):
                    if i in col_ids:
                        curr_line.append(float(line[i].strip()))
                dataset_trimmed.append(curr_line)
            dataset = dataset_trimmed
            print(col_ids)

        # Split data in train, val, test
        train_dataset, test_val_dataset, train_labels, test_val_labels = train_test_split(dataset, labels, test_size=0.2, random_state=1)
        val_dataset, test_dataset, val_labels, test_labels = train_test_split(test_val_dataset, test_val_labels, test_size=0.5, random_state=1)
        
        # Perform label balancing for training set
        oversample = SMOTE()
        train_dataset, train_labels = oversample.fit_resample(train_dataset, train_labels)
        train_data_and_labels = [[data, label] for data, label in zip(train_dataset, train_labels)]
        val_data_and_labels = [[data, label] for data, label in zip(val_dataset, val_labels)]
        test_data_and_labels = [[data, label] for data, label in zip(test_dataset, test_labels)]

        # Perform kmeans
        test = False
        k = 2
        if test:
            output = kmeans(train_data_and_labels, test_data_and_labels, False, k)
            output_baseline = kmeans(train_data_and_labels, test_data_and_labels, True, k)
        else:
            output = kmeans(train_data_and_labels, val_data_and_labels, False, k)
            output_baseline = kmeans(train_data_and_labels, val_data_and_labels, True, k)

        # Print metrics
        print_metrics(output)
        print_metrics(output_baseline)