# -*- compile-command: "python -m nes.control "; default-directory: "~/Projects/mechlearn/fami/" -*-

import zmq


def has_control(socket, start_state, default_seq):
    socket.send_json({"state": start_state,
                      "data": ["framebuffer"],
                      "inputs": default_seq,
                      "lastn": 1})
    result = socket.recv_json()
    fb_default = result["data"][-1]["framebuffer"]
    # TODO: some way to make requests in parallel? dealer on this side, router
    # on the other side?
    in_seqs = map(lambda i: [1 << i, 0] * (len(default_seq) / 2),
                  [0, 1, 4, 5, 6, 7])
    for move_seq in in_seqs:
        socket.send_json({"state": start_state,
                          "data": ["framebuffer"],
                          "inputs": move_seq,
                          "lastn": 1})
        result = socket.recv_json()
        if result["data"][-1]["framebuffer"] != fb_default:
            return True, move_seq
    return False, None


def start(services):
    context = zmq.Context()
    emusocket = context.socket(zmq.REQ)
    emusocket.connect("tcp://127.0.0.1:" + services["fceux"])
    ctrlsocket = context.socket(zmq.REP)
    ctrlsocket.bind("tcp://127.0.0.1:" + services["control"])
    # Wait for requests like "do I have control at this time point given this
    # optional default future"?
    while True:
        result = ctrlsocket.recv_json()
        state = result["state"]
        future = result.get("future", None)
        horizon = result.get("horizon", None)
        if horizon is not None and future is not None:
            future = future[:horizon * 2]
        elif horizon is not None and future is None:
            future = [0, 0] * horizon
        elif horizon is None and future is not None:
            horizon = len(future) / 2
        elif horizon is None and future is None:
            horizon = 5
            future = [0, 0] * horizon
        ret, moves = has_control(emusocket, state, future)
        ctrlsocket.send_json({"state": state,
                              "future": future,
                              "has_control": ret,
                              "control_moves": moves})


if __name__ == "__main__":
    import sys
    import json
    start(json.load(open("services.json", 'r')
                    if len(sys.argv) < 2 else sys.argv[1]))
