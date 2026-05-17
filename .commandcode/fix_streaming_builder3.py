"""Fix streaming builder to skip fp8 scale tensors in block key map.

input_scale and weight_scale are fp8 quantization metadata stored in the
checkpoint state dict. They get handled by SDOps/ModuleOps during model building,
NOT as separate parameters in the meta model architecture. Filter them out of
the block_key_map so the streaming builder doesn't try to look them up in the
meta model's named_parameters().
"""
path = "/home/eric-reitsma/LTX-2/packages/ltx-core/src/ltx_core/block_streaming/builder.py"

# Fix 1: filter input_scale/weight_scale from block_key_map in _scan_checkpoint_keys
old_scan = (
    '                    block_key_map.setdefault(block_idx, []).append((sft_key, param_name))'
)

new_scan = (
    '                    # Skip fp8 quantization scale tensors -- they are not parameters\n'
    '                    # in the meta model architecture and get handled by SDOps/ModuleOps\n'
    '                    # during state dict loading.\n'
    '                    if param_name.endswith(".input_scale") or param_name.endswith(".weight_scale"):\n'
    '                        non_block_keys.append((sft_key, model_key))\n'
    '                    else:\n'
    '                        block_key_map.setdefault(block_idx, []).append((sft_key, param_name))'
)

src = open(path).read()
if old_scan in src:
    src = src.replace(old_scan, new_scan, 1)
    open(path, "w").write(src)
    print("Fixed _scan_checkpoint_keys - skips input_scale/weight_scale in block_key_map")
else:
    print("Pattern for _scan_checkpoint_keys not found")

# Fix 2: revert the _build_pinned_source back to simple param lookup since
# scale tensors are no longer in block_key_map
old_build = (
    "            block_params = dict(blocks[block_idx].named_parameters())\n"
    "            block_buffers = dict(blocks[block_idx].named_buffers())\n"
    "            for _sft_key, param_name in entries:\n"
    '                key = make_block_key(self.blocks_prefix, block_idx, param_name)\n'
    "                if param_name in block_params:\n"
    "                    block_tensors[key] = block_params[param_name]\n"
    "                elif param_name in block_buffers:\n"
    "                    block_tensors[key] = block_buffers[param_name]\n"
    "                else:\n"
    '                    raise KeyError(f"{param_name} not found in parameters or buffers")'
)

new_build = (
    "            block_params = dict(blocks[block_idx].named_parameters())\n"
    "            for _sft_key, param_name in entries:\n"
    '                key = make_block_key(self.blocks_prefix, block_idx, param_name)\n'
    "                block_tensors[key] = block_params[param_name]"
)

src = open(path).read()
if old_build in src:
    src = src.replace(old_build, new_build, 1)
    open(path, "w").write(src)
    print("Reverted _build_pinned_source to simple param lookup")
else:
    print("Pattern for _build_pinned_source not found")

# Fix 3: revert _build_disk_source layout derivation
old_layout = (
    "        block_params = dict(blocks[0].named_parameters())\n"
    "        block_buffers = dict(blocks[0].named_buffers())\n"
    "        layout = derive_layout({**block_buffers, **block_params}, dtype)"
)

new_layout = (
    "        layout = derive_layout(dict(blocks[0].named_parameters()), dtype)"
)

src = open(path).read()
if old_layout in src:
    src = src.replace(old_layout, new_layout, 1)
    open(path, "w").write(src)
    print("Reverted _build_disk_source layout derivation")
else:
    print("Pattern for _build_disk_source layout not found")
