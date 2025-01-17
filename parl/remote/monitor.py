#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pickle
import random
import time
import zmq
import threading

from flask import Flask, render_template, jsonify

app = Flask(__name__)


@app.route('/')
@app.route('/workers')
def worker():
    return render_template('workers.html')


@app.route('/clients')
def clients():
    return render_template('clients.html')


class ClusterMonitor(object):
    """A monitor which requests the cluster status every 10 seconds.
    """

    def __init__(self, master_address):
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.socket.connect('tcp://{}'.format(master_address))
        self.data = None

        thread = threading.Thread(target=self.run)
        thread.setDaemon(True)
        thread.start()

    def run(self):
        master_is_alive = True
        while master_is_alive:
            try:
                self.socket.send_multipart([b'[MONITOR]'])
                msg = self.socket.recv_multipart()

                status = pickle.loads(msg[1])
                data = {'workers': [], 'clients': []}
                master_idx = None
                for idx, worker in enumerate(status['workers'].values()):
                    worker['load_time'] = list(worker['load_time'])
                    worker['load_value'] = list(worker['load_value'])
                    if worker['hostname'] == 'Master':
                        master_idx = idx
                    data['workers'].append(worker)
                if master_idx != 0 and master_idx is not None:
                    master_worker = data['workers'].pop(master_idx)
                    data['workers'] = [master_worker] + data['workers']

                data['clients'] = list(status['clients'].values())
                self.data = data
                time.sleep(10)

            except zmq.error.Again as e:
                master_is_alive = False
                self.socket.close(0)

    def get_data(self):
        assert self.data is not None
        return self.data


@app.route('/cluster')
def cluster():
    data = CLUSTER_MONITOR.get_data()
    return jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor_port', default=1234, type=int)
    parser.add_argument('--address', default='localhost:8010', type=str)
    args = parser.parse_args()

    CLUSTER_MONITOR = ClusterMonitor(args.address)
    app.run(host="0.0.0.0", port=args.monitor_port)
