"""Fix streaming builder to also look up named_buffers() for fp8 input_scale/weight_scale."""
path = "/home/eric-reitsma/LTX-2/packages/ltx-core/src/ltx_core/block_streaming/builder.py"

old = (
    "        blocks = resolve_attr(meta_model, self.blocks_attr)\n"
    "        block_tensors: dict[str, torch.Tensor] = {}\n"
    "        for block_idx, entries in block_key_map.items():\n"
    "            block_params = dict(blocks[block_idx].named_parameters())\n"
    "            for _sft_key, param_name in entries:\n"
    '                key = make_block_key(self.blocks_prefix, block_idx, param_name)\n'
    "                block_tensors[key] = block_params[param_name]"
)

new = (
    "        blocks = resolve_attr(meta_model, self.blocks_attr)\n"
    "        block_tensors: dict[str, torch.Tensor] = {}\n"
    "        for block_idx, entries in block_key_map.items():\n"
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

src = open(path).read()
if old in src:
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print("Fixed _build_pinned_source - includes named_buffers() lookup")
else:
    print("Pattern not found in builder.py")
