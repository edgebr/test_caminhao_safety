from live_graph import VectorGraph, LiveGraph
import matplotlib.pyplot as plt
from thread_util import run_async
import queue
import serial
import math
import sys
import traceback
from termcolor import colored

acc = queue.Queue()
gyro = queue.Queue()
acc_filtered = queue.Queue()
gyro_filtered = queue.Queue()


def apppend_to_queues(queues: dict, line: bytes):
    try:
        datas = line.decode('utf-8')
        datas = datas[:-2].split(',')

        queues['acc'].put((datas[0], datas[1], datas[2]))
        queues['gyro'].put((datas[3], datas[4], datas[5]))
        queues['acc_filtered'].put((datas[6], datas[7], datas[8]))
        queues['gyro_filtered'].put((datas[9], datas[10], datas[11]))
    except UnicodeDecodeError:
        pass


@run_async
def safety(queues: dict, port: str, file_handler):
    ser = serial.Serial(port=port, baudrate=115200)
    line = b''
    while True:
        line = ser.readline()
        if b'#' == line[0:1]:
            if line[0:2] == b'#@':
                print(colored(f'{line[2:-2].decode("utf-8")}', 'blue', attrs=['bold']))
            elif line[0:2] == b'#&':
                print(colored(f'{line[2:-2].decode("utf-8")}', 'yellow', attrs=['dark', 'bold']))
            elif line[0:2] == b'#*':
                print(colored(f'{line[2:-2].decode("utf-8")}', 'green', attrs=['bold']))
            elif line[0:2] == b'#?':
                print(colored(f'{line[2:-2].decode("utf-8")}', 'red', attrs=['bold']))
            else:
                try:
                    file_handler.write(line[1:].decode('utf-8'))
                except ValueError:
                    return
                apppend_to_queues(queues, line[1:])
        elif b'$' == line[0:1]:
            print("Error BNO080")
            line = b''
        else:
            line = b''


def get_acc_data(frame):
    data_vec = acc.get()
    return (float(data_vec[0]), float(data_vec[1]), float(data_vec[2]))


def get_gyro_data(frame):
    data_vec = gyro.get()
    return (float(data_vec[0]), float(data_vec[1]), float(data_vec[2]))


def get_acc_filtered_data(frame):
    data_vec = acc_filtered.get()
    return (float(data_vec[0]), float(data_vec[1]), float(data_vec[2]))


def get_gyro_filtered_data(frame):
    data_vec = gyro_filtered.get()
    return (float(data_vec[0]), float(data_vec[1]), float(data_vec[2]))


try:
    csv_filename = sys.argv[1]
    csv_file = open(csv_filename + '.csv', 'w')
    csv_file.write('acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, f_acc_x, f_acc_y, f_acc_z, f_gyro_x, f_gyro_y, f_gyro_z\n')

    def handle_key(event):
        global csv_file
        sys.stdout.flush()

        if event.key == 'ctrl+c':
            csv_file.close()
            print('closed')
            exit(0)

    task1 = safety({
        'acc': acc,
        'gyro': gyro,
        'acc_filtered': acc_filtered,
        'gyro_filtered': gyro_filtered,
    }, sys.argv[2], csv_file)

    window = 10
    # acc_graph = VectorGraph('Aceleracao', window, -1.2, 1.2, 100, get_acc_data)
    # gyro_graph = VectorGraph('Giroscopio', window, -1, 1, 100, get_gyro_data)
    acc_name = 'Aceleracao - ' + (sys.argv[3] if len(sys.argv) >= 4 else '')
    gyro_name = 'Giroscopio - ' + (sys.argv[3] if len(sys.argv) >= 4 else '')
    acc_filtered_graph = VectorGraph(acc_name, window, -1.2, 1.2, 100, get_acc_filtered_data)
    gyro_filtered_graph = VectorGraph(gyro_name, window, -1, 1, 100, get_gyro_filtered_data)

    acc_filtered_graph.fig.canvas.mpl_connect('key_press_event', handle_key)
    plt.show()
except Exception as e:
    print(traceback.format_exc())
finally:
    csv_file.close()
