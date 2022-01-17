epochs=2
rm -rf cc_scoring && ./execute_train_test.sh --clients 8 -m SLTrainer $epochs exp3KmeansCustom $epochs rl $epochs srl $epochs -d && ./execute_train_test.sh --clients 8 -m sl srl rl exp3KmeansCustom -d -ts
