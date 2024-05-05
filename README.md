# Spark Scheduler Comparison and Algorithm Performance Analysis

This repository contains an analysis of different Spark schedulers and the performance of various algorithms implemented in Spark MLlib. We conducted experiments using different Spark schedulers including FIFO and Fair, and measured the execution time of algorithms such as FP Growth and Random Forest. Additionally, we incorporated hyperparameter tuning for these algorithms using grid search cross-validation.

## Overview

In distributed computing frameworks like Spark, efficient job scheduling is crucial for optimizing resource utilization and improving overall performance. Spark provides different schedulers, each with its own scheduling policies and strategies. Understanding how these schedulers impact job execution time and resource allocation can provide valuable insights for optimizing Spark applications.

Moreover, the choice of algorithm and its parameter settings can significantly affect the performance of machine learning tasks. In this analysis, we focused on popular algorithms available in Spark MLlib and evaluated their performance under different scheduling configurations.

## Model Architecture



## Experiments

### Spark Schedulers

1. **FIFO Scheduler**: The First-In-First-Out (FIFO) scheduler is the default scheduler in Spark. It schedules jobs in the order they are submitted to the cluster, without considering resource availability or job priority.
   
2. **Fair Scheduler**: The Fair Scheduler aims to provide fair distribution of resources among users and applications. It dynamically adjusts resource allocation based on the demand from different jobs or users.

### Algorithms

1. **FP Growth**: FP Growth is a frequent pattern mining algorithm used for mining frequent itemsets from transaction data efficiently. We evaluated its performance under different scheduling configurations.

2. **Random Forest**: Random Forest is an ensemble learning method used for classification and regression tasks. We measured its execution time and accuracy under various scheduling settings.

### Hyperparameter Tuning

Hyperparameter tuning is essential for optimizing the performance of machine learning models. We employed grid search cross-validation to find the best combination of hyperparameters for FP Growth and Random Forest algorithms.

## Cluster Architecture

Our Spark cluster consists of multiple nodes, with one master node and several worker nodes. When a job is submitted to the cluster, it is divided into tasks, which are then distributed among the worker nodes for parallel execution. Once the tasks are completed, the results are aggregated by the master node, which then presents the final results to the user.
