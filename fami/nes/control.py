import zmq


def has_control(socket, start_state, default_seq, horizon=3):
    if default_seq is None:
        default_seq = [0, 0] * horizon
    socket.send_json({"state": start_state,
                      "data": ["framebuffer"],
                      "inputs": default_seq[:horizon * 2],
                      "lastn": 1})
    result = socket.recv_json()
    fb_default = result["data"][-1]["framebuffer"]
    # TODO: some way to make requests in parallel? dealer on this side, router
    # on the other side?
    in_seqs = map(lambda i: [1 << i, 0] * horizon, [0, 1, 4, 5, 6, 7])
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
        ret, moves = has_control(emusocket, state, future, horizon)
        ctrlsocket.send_json({"state": state,
                              "future": future,
                              "horizon": horizon,
                              "has_control": ret,
                              "control_moves": moves})


if __name__ == "__main__":
    import sys
    import json
    start(json.load(open("services.json", 'r')
                    if len(sys.argv) < 2 else sys.argv[1]))
