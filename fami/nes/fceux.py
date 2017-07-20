import fceulib
import zmq
import datrie
import collections
import string
import logging
import timeit


def start(config):
    rom = config.get("rom", "mario.nes")
    amap = datrie.AlphaMap()
    amap.add_range("0", "9")
    amap.add_range("a", "f")
    inputs_to_keys = datrie.Trie(alpha_map=amap)
    #states_to_keys = {}
    keys_to_states = {}
    emu = fceulib.runGame(rom)
    init_state = fceulib.VectorBytes()
    emu.save(init_state)
    #states_to_keys[tuple(init_state)] = 0
    keys_to_states[0] = (init_state, u"")
    # No need to put something in inputs_to_keys for null string
    next_state = 1

    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:" + config["fceux"])

    while True:
        msg = sock.recv_json()
        tstart = timeit.default_timer()
        # msg will be "from state id SID, do these inputs as a sequence of
        # numbers"
        start_id = msg["state"]
        inputs = msg["inputs"]
        logging.info("Start; asking for %d net states" % (len(inputs) / 2))
        # print "hi", inputs_to_keys.items(), map(lambda a:
        # keys_to_states[a][1], keys_to_states)
        wanted_data = msg.get("data", [])
        num_interesting = msg.get("lastn", None)
        interesting = collections.deque(maxlen=num_interesting)
        assert start_id in keys_to_states
        state, prefix = keys_to_states[start_id]
        input_key = prefix
        for i in inputs:
            # Convert to hex
            input_key += hex(i)[2:].zfill(2)
        longest_prefix, longest_id = inputs_to_keys.longest_prefix_item(
            input_key,
            (u"", 0)
        )
        # The start state is interesting
        interesting.append(start_id)
        # Following states are also interesting.
        # We handle the start and following separately in case we start at t=0
        # We also iterate up to longest_prefix+4 because we want to include
        # that found prefix.
        # (4 chars per time step: 2 for p1, 2 for p2 since we use a hex encoding)
        for i in range(len(prefix) + 4, len(longest_prefix) + 4, 4):
            segment = longest_prefix[:i]
            state_id = inputs_to_keys[segment]
            # print "Reuse", state_id, "from", segment
            interesting.append(state_id)
        remaining = (len(input_key) - len(longest_prefix)) / 2
        prefix_key = longest_prefix
        logging.info("Running %d new emulation steps" % (remaining / 2))
        temustart = timeit.default_timer()
        if remaining > 0:
            emu.load(keys_to_states[longest_id][0])
        for i in range(0, remaining, 2):
            inp = inputs[i]
            prefix_key = prefix_key + hex(inp)[2:].zfill(2)
            inp2 = inputs[i + 1]
            prefix_key += hex(inp2)[2:].zfill(2)
            here_state = fceulib.VectorBytes()
            emu.stepFull(inp, inp2)
            emu.save(here_state)
            # we know this is a brand new input sequence but it might be a symmetric state
            # this way we can track both of those possibilities
            # hst = tuple(here_state)
            if False:  # hst in states_to_keys:
                # state = states_to_keys[hst]
                # inputs_to_keys[prefix_key] = state
                # interesting.append(state)
                pass
            else:
                # states_to_keys[hst] = next_state
                keys_to_states[next_state] = here_state, prefix_key
                inputs_to_keys[prefix_key] = next_state
                interesting.append(next_state)
                next_state += 1
        tdone = timeit.default_timer()
        logging.info("Time to use cache: %0.6f; Time to emulate: %0.6f" %
                     (temustart - tstart, tdone - temustart))
        # Now, what do we gather up and send back?
        out_msg = {
            "states": list(interesting),
            "data": []
        }

        for i in interesting:
            here_data = {}
            state, prefix = keys_to_states[i]
            emu.load(state)
            for d in wanted_data:
                if d == "framebuffer":
                    here_data[d] = list(emu.image)
                elif d == "inputs":
                    input_moves = [string.atoi(prefix[ii:i + 2], 16)
                                   for ii in range(len(prefix))]
                    here_data[d] = (input_moves[::2], input_moves[1::2])
                elif d == "ram":
                    buf = emu.memory
                    here_data[d] = list(buf)
                elif d == "nta":
                    buf = emu.fc.ppu.NTARAM
                    here_data[d] = list(buf)
                elif d == "chr":
                    bufs = (emu.fc.ppu.getVNAPage(0),
                            emu.fc.ppu.getVNAPage(1))
                    here_data[d] = list(bufs[0]) + list(bufs[1])
                elif d == "pal":
                    buf = emu.fc.ppu.PALRAM
                    here_data[d] = list(buf)
                elif d == "oam":
                    buf = emu.fc.ppu.SPRAM
                    here_data[d] = list(buf)
                elif d == "ppu":
                    buf = emu.fc.ppu.values
                    here_data[d] = list(buf)
                else:
                    assert False, "Unrecognized data request " + d
            out_msg["data"].append(here_data)
        sock.send_json(out_msg)


if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)
    start(json.load(open("services.json", 'r'))
          if len(sys.argv) < 2 else sys.argv[1])

# Next steps:
# Child worker processes
