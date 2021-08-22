from posix import times_result
import time
import os
import subprocess
import signal
import argparse
import yaml

SETTINGS_OFFLINE = "./src/settings_offline.yml"

EXP3_WEIGHTS_DIR = "./weights/exp3"
DELETE_EXP3_WEIGHTS_DIR = "rm -r " + EXP3_WEIGHTS_DIR

PSQL_CONNECT = 'psql "host=127.0.0.1 port=5432 dbname=puffer user=puffer password=123456"'
GET_EXPERIMENTS = 'SELECT * FROM puffer_experiment;'
DELETE_EXPERIMENT = "delete from puffer_experiment where id='{}';"

INFLUX = "influx"
INFLUX_USE_PUFFER = "use puffer"
DELETE_BY_ID = "delete from active_streams, client_buffer, client_sysinfo, video_acked, video_sent where expt_id='{}'"


RUN_OFFLINE_TEST = "python3 ./src/media-server/offline_test.py"
RUN_EXP3_SERVER = "python3.7 ./src/scripts/exp3_server.py"
CLUSTER = "python3.7 ./src/scripts/clustering.py -f"


def update_yaml_settings(delta, clusters):
    with open(SETTINGS_OFFLINE, 'r') as f:
        yaml_settings = yaml.safe_load(f)

    with open(SETTINGS_OFFLINE, 'w') as f:
        yaml_settings["experiments"][0]['fingerprint']['abr_config']['training_mode'] = True
        yaml_settings["experiments"][0]['fingerprint']['abr_config']['delta'] = delta
        yaml_settings["experiments"][0]['fingerprint']['abr_config']['clusters'] = clusters

        yaml.safe_dump(yaml_settings, f)


def train(slim_train=True):
    print('start training')

    subprocess.check_call(CLUSTER, shell=True)
    print('finish clustering')

    if os.path.isdir(EXP3_WEIGHTS_DIR):
        subprocess.check_call(DELETE_EXP3_WEIGHTS_DIR, shell=True)

    exp3_server_log = open("exp3_server_logs.txt", 'w')
    offline_test_log = open("offline_train_logs.txt", 'w')

    exp3_server = subprocess.Popen(
        RUN_EXP3_SERVER, shell=True, stdout=exp3_server_log, preexec_fn=os.setpgrp)
    time.sleep(5)

    offline_test_command = RUN_OFFLINE_TEST
    if not slim_train:
        offline_test_command += ' --epochs 3 --num-of-traces 300'

    offline_test = subprocess.Popen(
        offline_test_command, shell=True, stdout=offline_test_log)

    offline_test.wait()
    
    os.killpg(os.getpgid(exp3_server.pid), signal.SIGTERM)

    time.sleep(5)

    exp3_server_log.close()
    offline_test_log.close()

    print('finish running experiment')

    # delete logs
    # psql = subprocess.Popen(PSQL_CONNECT, shell=True,
    #                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    # psql.stdin.write(GET_EXPERIMENTS)
    # psql.stdin.close()
    # psql_experiment = psql.stdout.readlines()[-3]

    # assert '"training_mode": "true"' in psql_experiment
    # args = psql_experiment.split('|')
    # training_expt_id = args[0].strip()

    # print(f'delete experiment: "{psql_experiment}"')

    # # delete from influx
    # psql = subprocess.Popen(INFLUX, shell=True,
    #                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    # psql.stdin.write(INFLUX_USE_PUFFER)
    # psql.stdin.write(DELETE_BY_ID.format(training_expt_id))
    # psql.stdin.close()

    print('end training')


def test():
    with open(SETTINGS_OFFLINE, 'r') as f:
        yaml_settings = yaml.safe_load(f)

    with open(SETTINGS_OFFLINE, 'w') as f:
        yaml_settings["experiments"][0]['fingerprint']['abr_config']['training_mode'] = False
        yaml.safe_dump(yaml_settings, f)

    print('start training')

    offline_test_log = open("offline_test_logs.txt", 'w')

    offline_test = subprocess.Popen(
        RUN_OFFLINE_TEST, shell=True, stdout=offline_test_log)

    offline_test.wait()

    offline_test_log.close()

    print('end testing')


def main():
    deltas = [0.9]
    clusters = [8]

    for delta in deltas:
        for cluster in clusters:
            print(f'running exp with DELTA={delta}, CLUSTERS={cluster}')
            update_yaml_settings(delta, cluster)

            slim_train = True
            if cluster > 8:
                slim_train = False

            train(slim_train)
            test()


if __name__ == "__main__":
    main()
