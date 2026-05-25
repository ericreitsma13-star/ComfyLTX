#!/usr/bin/env python3
"""Generate full ComfyUI JSON with node positions from the API format."""
import json

with open("/home/ericr/ComfyUI/workflow_i2v_audio_api.json") as f:
    api = json.load(f)

nodes = []
links = []
link_id = 1
positions = {
    10: [0, 0], 11: [0, 200], 12: [0, 400], 13: [0, 600],
    20: [400, 0], 30: [400, 500], 31: [400, 700], 32: [700, 600],
    40: [400, 200], 41: [400, 900],
    42: [700, 900], 43: [700, 300], 44: [1000, 250],
    46: [1300, 600], 47: [1600, 500],
    50: [700, 0], 51: [700, 100], 52: [1000, 0], 53: [1300, 0],
    54: [1600, 0], 60: [1900, 400],
    61: [2200, 300], 62: [2200, 600],
    70: [2500, 300], 71: [2800, 300],
}
sizes = {
    10: [300, 100], 11: [300, 100], 12: [300, 100], 13: [400, 150],
    20: [350, 100], 30: [400, 200], 31: [400, 200], 32: [300, 100],
    40: [300, 100], 41: [300, 100], 42: [300, 100], 43: [350, 150],
    44: [400, 120], 46: [300, 100], 47: [500, 200],
    50: [300, 100], 51: [300, 80], 52: [400, 150], 53: [300, 100],
    54: [400, 120], 60: [300, 100], 61: [500, 150], 62: [300, 100],
    70: [300, 100], 71: [350, 150],
}

node_order = 0
for nid_str, node_data in api.items():
    nid = int(nid_str)
    ct = node_data["class_type"]
    inputs = node_data["inputs"]

    node_inputs = []
    slot = 0
    for iname, ival in inputs.items():
        if isinstance(ival, list):
            from_node, from_slot = ival
            node_inputs.append({
                "name": iname,
                "type": "any",
                "link": link_id,
            })
            links.append([link_id, from_node, from_slot, nid, slot, "any"])
            link_id += 1
            slot += 1

    node_outputs = []
    for other_nid_str, other_data in api.items():
        for oname, oval in other_data["inputs"].items():
            if isinstance(oval, list) and oval == [nid, 0]:
                node_outputs.append({"name": f"out0", "type": "any", "links": None})
                break
        break
    if not node_outputs:
        node_outputs.append({"name": "output", "type": "any", "links": None})

    widget_values = []
    for iname, ival in inputs.items():
        if not isinstance(ival, list):
            widget_values.append(ival)

    pos = positions.get(nid, [0, node_order * 150])
    sz = sizes.get(nid, [300, 100])

    n = {
        "id": nid,
        "type": ct,
        "pos": pos,
        "size": sz,
        "flags": {},
        "order": node_order,
        "mode": 0,
        "inputs": node_inputs,
        "outputs": node_outputs,
        "properties": {"Node name for S&R": ct},
        "widgets_values": widget_values,
    }
    nodes.append(n)
    node_order += 1

full = {
    "last_node_id": max(int(k) for k in api.keys()),
    "last_link_id": link_id - 1,
    "nodes": nodes,
    "links": links,
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}

with open("/home/ericr/ComfyUI/workflow_i2v_audio_full.json", "w") as f:
    json.dump(full, f, indent=2)
print(f"✅ workflow_i2v_audio_full.json — {len(nodes)} nodes, {len(links)} links")
