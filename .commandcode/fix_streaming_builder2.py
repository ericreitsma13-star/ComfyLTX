"""Fix _build_disk_source to also include buffers in layout."""
path = "/home/eric-reitsma/LTX-2/packages/ltx-core/src/ltx_core/block_streaming/builder.py"

old = (
    "        layout = derive_layout(dict(blocks[0].named_parameters()), dtype)"
)

new = (
    "        block_params = dict(blocks[0].named_parameters())\n"
    "        block_buffers = dict(blocks[0].named_buffers())\n"
    "        layout = derive_layout({**block_buffers, **block_params}, dtype)"
)

src = open(path).read()
if old in src:
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print("Fixed _build_disk_source layout derivation")
else:
    print("Pattern not found")
