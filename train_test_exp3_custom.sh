epochs=2
rm -rf cc_scoring && ./execute_train_test.sh --clients 8 -m SLTrainer $epochs exp3KmeansCustom $epochs -d && ./execute_train_test.sh --clients 8 -m constant_0 constant_1 constant_2 exp3KmeansCustom -d -ts
