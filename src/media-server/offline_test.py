#!/usr/bin/env python3
import time
import os
import subprocess
import signal

LOGS_FILE = './weights/logs.txt'

def run_offline_media_servers():
    run_server_html_cmd = 'python3 ./src/portal/manage.py runserver 0:8080'
    p1 = subprocess.Popen(run_server_html_cmd, shell=True)
    time.sleep(4)
    run_servers_cmd = './src/media-server/run_servers ./src/settings_offline.yml'
    p2 = subprocess.Popen(run_servers_cmd, shell=True)
    time.sleep(4)


def start_maimahi_clients(num_clients):
    logs_file = open(LOGS_FILE, 'w')
    plist = []
    try:
        trace_dir = "./traces/mahimahi"

        files = os.listdir(trace_dir)
        test_files = files[:300]
        reinforce_train_files = files[300:600]

        traces = test_files
        epochs = 50
        for epoch in range(epochs):
            # for filename in files[:300]:
            for f in range(0, len(traces), num_clients):
                logs_file.write(f"Epoch: {epoch}/{epochs}. Files: {f}/{len(traces)}\n")
                logs_file.flush()
                # mahimahi_cmd = 'mm-delay 40 mm-link 12mbps ' + trace_dir + '/' + \
                #                filename
                base_port = 9360
                remote_base_port = 9222
                plist = []
                for i in range(1, num_clients + 1):
                    filename = traces[i]
                    remote_port = remote_base_port + i
                    port = base_port + i
                    #chrome_cmd = 'chromium-browser --headless --disable-gpu --remote-debugging-port=9222 ' + \
                    #             'http://$MAHIMAHI_BASE:8080/player/?wsport=' + \
                    #             str(port) + ' --user-data-dir=./' + str(port) + \
                    #             '.profile'

                    time.sleep(4)
                    # mahimahi_chrome_cmd = "mm-delay 40 mm-link /home/csuser/puffer/src/media-server/12mbps {}/{} -- sh -c 'chromium --disable-gpu --remote-debugging-port={} http://100.64.0.1:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(trace_dir, filename, port, port, port)
                    mahimahi_chrome_cmd = "mm-delay 40 mm-link /home/ofir/puffer/src/media-server/12mbps {}/{} -- sh -c 'chromium --disable-gpu --headless --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(trace_dir, filename, remote_port, port, port)
                    # mahimahi_chrome_cmd = "mm-delay 40 mm-link /home/ofir/puffer/src/media-server/12mbps {}/{} -- sh -c 'chromium-browser --disable-gpu --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(trace_dir, filename, remote_port, port, port)
                    # print(mahimahi_chrome_cmd)
                    chrome_cmd_b = mahimahi_chrome_cmd.encode('utf-8')
                    p = subprocess.Popen(mahimahi_chrome_cmd, shell=True,
                                        preexec_fn=os.setsid)
                    plist.append(p)

                time.sleep(60*5)
                for p in plist:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(4)

                subprocess.check_call("rm -rf ./*.profile", shell=True,
                                    executable='/bin/bash')
    except Exception as e:
        print("exception: " + str(e))
        pass
    finally:
        logs_file.close()
        for p in plist:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            subprocess.check_call("rm -rf ./*.profile", shell=True,
                                  executable='/bin/bash')


def main():
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    run_offline_media_servers()
    time.sleep(6000)
    # start_maimahi_clients(1)


if __name__ == '__main__':
    main()
