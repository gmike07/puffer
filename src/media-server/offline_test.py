#!/usr/bin/env python3
import time
import os
import subprocess
import signal

EPOCHS = 1
NUM_OF_CLIENTS = 10
LOGS_FILE = './weights/logs.txt'

BASE_PORT = 9360
REMOTE_BASE_PORT = 9222


def run_offline_media_servers():
    run_server_html_cmd = 'python3 ./src/portal/manage.py runserver 0:8080'
    p1 = subprocess.Popen(run_server_html_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(3)
    run_servers_cmd = './src/media-server/run_servers ./src/settings_offline.yml'
    p2 = subprocess.Popen(run_servers_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    return p1, p2


def start_maimahi_clients(num_clients):
    logs_file = open(LOGS_FILE, 'w')
    plist = []
    try:
        trace_dir = "./traces/mahimahi"
        files = os.listdir(trace_dir)
        test_files = files[:1400]

        traces = test_files
        for epoch in range(EPOCHS):
            for f in range(0, len(traces), num_clients):
                logs_file.write(
                    f"Epoch: {epoch}/{EPOCHS}. Files: {f}/{len(traces)}\n")
                logs_file.flush()

                p1, p2 = run_offline_media_servers()
                plist = [p1, p2]

                for i in range(1, num_clients + 1):
                    filename = traces[i]
                    remote_port = REMOTE_BASE_PORT + i
                    port = BASE_PORT + i

                    time.sleep(2)
                    mahimahi_chrome_cmd = "mm-delay 40 mm-link /home/ubuntu/puffer/src/media-server/12mbps {}/{} -- sh -c 'chromium disable-infobars --disable-gpu --headless --enable-logging=true --v=1 --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(
                        trace_dir, filename, remote_port, port, port)

                    p = subprocess.Popen(mahimahi_chrome_cmd, shell=True,
                                         preexec_fn=os.setsid)
                    plist.append(p)

                time.sleep(60*10)
                for p in plist:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(4)

                subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
    except Exception as e:
        print("exception: " + str(e))
    finally:
        logs_file.close()
        for p in plist:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            subprocess.check_call("rm -rf ./*.profile", shell=True,
                                  executable='/bin/bash')


def main():
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    # run_offline_media_servers()
    start_maimahi_clients(NUM_OF_CLIENTS)


if __name__ == '__main__':
    main()
