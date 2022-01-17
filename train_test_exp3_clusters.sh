epochs=2
rm -rf cc_scoring && ./execute_train_test.sh --clients 8 -m SLTrainer $epochs exp3KmeansCustom $epochs contextlessClusterTrainer 1 contextlessExp3Kmeans $epochs SLClusterTrainer 1 exp3Kmeans $epochs -d && ./execute_train_test.sh --clients 8 -m exp3Kmeans contextlessExp3Kmeans exp3KmeansCustom -d -ts
