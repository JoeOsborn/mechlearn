import fceulib
import numpy as np
from fceulib import VectorBytes
import sys
import zmq
import time


def send_msg(socket, extra_data, A, flags=0, copy=True, track=False):
    md = dict(
        _dtype=str(A.dtype),
        _shape=A.shape,
    )
    md.update(extra_data)
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def recv_msg(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['_dtype'])
    return md, A.reshape(md['_shape'])


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

rom = sys.argv[1] if len(sys.argv) > 1 else 'mario.nes'
emu = fceulib.runGame(rom)
init_state = VectorBytes()
emu.save(init_state)

# later, put this in a database
ist = tuple(init_state)
states_to_keys = {ist: 0}
keys_to_states = {0: init_state}

while True:
    #  Wait for next request from client
    message, inputs = recv_msg(socket)
    start_state = message["state"]
    assert start_state in keys_to_states
    emu.load(keys_to_states[start_state])
    seen_states = [start_state]
    for i1, i2 in zip(inputs[:, 0], inputs[:, 1]):
        here_state = VectorBytes()
        emu.stepFull(i1, i2)
        emu.save(here_state)
        hst = tuple(here_state)
        if hst not in states_to_keys:
            key = len(states_to_keys)
            states_to_keys[hst] = key
            keys_to_states[key] = here_state
        seen_states.append(states_to_keys[hst])
    #  Send reply back to client
    socket.send_json({
        "states": seen_states
    })
