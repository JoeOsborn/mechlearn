# -*- compile-command: "python -m nes.fceux "; default-directory: "~/Projects/mechlearn/fami/" -*-

import fceulib
import zmq
import datrie
import collections
import string
import logging
import timeit
import numpy


def last_move(prefix):
    in1 = int(prefix[-4:-2], 16)
    in2 = int(prefix[-2:], 16)
    return in1, in2


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

    # TODO: use a hash of the state as the key?
    # It feels off that it is so sensitive to ordering.

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
        # print "MSG:", msg
        inputs = msg.get("inputs", [])
        logging.info("Start; asking for %d net states" % (len(inputs) / 2))
        if start_id == 0 and len(inputs) == 0:
            logging.warning(
                "Data about just the initial state is probably not going to be interesting or useful")
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
            start_save, start_save_prefix = keys_to_states[longest_id]
            emu.load(start_save)
            if len(start_save_prefix) > 0:
                start_in1, start_in2 = last_move(start_save_prefix)
                emu.stepFull(start_in1, start_in2)
        for i in range(0, remaining, 2):
            inp = inputs[i]
            prefix_key = prefix_key + hex(inp)[2:].zfill(2)
            inp2 = inputs[i + 1]
            prefix_key += hex(inp2)[2:].zfill(2)
            here_state = fceulib.VectorBytes()
            # We save the state just beforehand, so we can load it up and then
            # do a stepFull to be sure all the emulator state is correct later
            # on.
            emu.save(here_state)
            emu.stepFull(inp, inp2)
            # we know this is a brand new input sequence but it might be a symmetric state
            # this way we can track both of those possibilities
            #hst = tuple(here_state)
            if False:  # hst in states_to_keys:
                #    state = states_to_keys[hst]
                #    inputs_to_keys[prefix_key] = state
                #    interesting.append(state)
                pass
            else:
                #states_to_keys[hst] = next_state
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
        img = fceulib.VectorBytes()
        for i in interesting:
            # TODO: often, interesting is a linear sequence so nearly
            # all of the loads are wasted.  Can that be avoided?
            here_data = {}
            # UGH UGH UGH
            # have to actually load up one frame before state, then click it
            # forward by the right inputs
            state, prefix = keys_to_states[i]
            emu.load(state)
            if len(prefix) > 0:
                in1, in2 = last_move(prefix)
                emu.stepFull(in1, in2)
            for d in wanted_data:
                if d == "framebuffer":
                    emu.imageInto(img)
                    here_data[d] = (numpy.array(img, copy=False).reshape(
                        (256, 256, 4)) / 255.).tolist()
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
                elif d == "mirror":
                    here_data[d] = emu.fc.cart.mirroring
                elif d == "chr":
                    # bufs = (emu.fc.ppu.getVNAPage(0),
                    #         emu.fc.ppu.getVNAPage(1))
                    #here_data[d] = list(bufs[0]) + list(bufs[1])
                    buf1 = numpy.array(
                        emu.fc.cart.getVPageChunk(0),
                        copy=False,
                        dtype=numpy.uint8)[:0x1000]
                    buf2 = numpy.array(
                        emu.fc.cart.getVPageChunk(0x1000),
                        copy=False,
                        dtype=numpy.uint8)[:0x1000]
                    here_data[d] = buf1.tolist() + buf2.tolist()
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
