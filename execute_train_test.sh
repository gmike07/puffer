sudo sysctl -w fs.inotify.max_user_instances=100000
sudo sysctl -w fs.inotify.max_user_watches=100000
sudo sysctl -p
echo $@
python3 ./src/media-server/train_test_simulation_main.py $@
