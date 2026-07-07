import json, requests, sys, random

API = "http://127.0.0.1:8188/prompt"

# Load the official workflow JSON
with open('/home/ericr/snap/code/238/.local/share/opencode/tool-output/tool_e4c566497001dOZThCzs9yeLfn') as f:
    workflow = json.load(f)

# Extract the subgraph nodes (the actual internal nodes)
subgraph = workflow['definitions']['subgraphs'][0]
nodes = subgraph['nodes']
links = subgraph['links']
inputs = subgraph['inputs']

# Convert to API format
prompt = {}
for node in nodes:
    nid = str(node['id'])
    prompt[nid] = {
        "class_type": node['type'],
        "inputs": {}
    }
    for inp in node.get('inputs', []):
        inp_name = inp['name']
        link_id = inp.get('link')
        if link_id is not None:
            # Find the link definition
            for link in links:
                if link['id'] == link_id:
                    origin_id = link['origin_id']
                    origin_slot = link['origin_slot']
                    prompt[nid]['inputs'][inp_name] = (str(origin_id), origin_slot)
                    break
        else:
            # Widget-based input - use widget value if available
            pass

    # Widget values
    if 'widgets_values' in node and node['widgets_values']:
        widgets = node.get('properties', {}).get('widgets', {})
        widget_names = list(node.get('properties', {}).get('inputOrder', {}).get('required', []))
        if not widget_names:
            # Try to extract from proxyWidgets
            pass
        wv = node['widgets_values']
        # Get input names from the class definition
        # For now, skip widget-only inputs since they'll be defaults

# Set the text prompt, dimensions, etc.
# The grouped node proxies widget values to internal nodes
# We need to set the values on the grouped node (128) which proxies to internal nodes
for node in workflow['nodes']:
    if node['id'] == 128:  # The grouped node
        prompt['128'] = {
            "class_type": node['type'],
            "inputs": {}
        }
        for inp in node.get('inputs', []):
            prompt['128']['inputs'][inp['name']] = inp.get('widget', {}).get('value')

# This approach is getting complex. Let me try a simpler way.
# Instead of flattening the subgraph, let me just use the top-level JSON structure.
# ComfyUI API can accept workflow JSON format directly.

# Actually the simplest approach: just extract subgraph nodes and submit directly
# with all internal links resolved.

# Let me take a different approach entirely:
# Use the official workflow as-is with the grouped node, setting widget proxies.

# First, build the API prompt from the top-level nodes (not the subgraph)
api_prompt = {}
for node in workflow['nodes']:
    nid = str(node['id'])
    api_prompt[nid] = {
        "class_type": node['type'],
        "inputs": {}
    }
    for inp in node.get('inputs', []):
        inp_name = inp['name']
        link_id = inp.get('link')
        if link_id is not None:
            # Top-level links (the simplified ones)
            for link in workflow['links']:
                if link[0] == link_id:
                    origin_id = link[1]
                    origin_slot = link[3]
                    api_prompt[nid]['inputs'][inp_name] = (str(origin_id), origin_slot)
                    break

# Set widget values on the grouped node (which proxies to subgraph nodes)
# For node 128, set the proxy widget values
api_prompt['128']['inputs']['unet_name'] = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
api_prompt['128']['inputs']['unet_name_1'] = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
api_prompt['128']['inputs']['clip_name'] = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
api_prompt['128']['inputs']['vae_name'] = "wan_2.1_vae.safetensors"
api_prompt['128']['inputs']['value'] = 5.0  # duration
api_prompt['128']['inputs']['value_1'] = False  # enable_turbo_mode
api_prompt['128']['inputs']['lora_name'] = "None"
api_prompt['128']['inputs']['lora_name_1'] = "None"

# Set text prompt via the grouped node's STRING input
# The grouped node has a text input that maps to the CLIPTextEncode widget

# Save video node
api_prompt['80']['inputs']['filename_prefix'] = "wan22_official"
api_prompt['80']['inputs']['format'] = "auto"
api_prompt['80']['inputs']['codec'] = "auto"

# Set the positive prompt - it should proxy through to the CLIPTextEncode
# In the subgraph input list, text input id=72e77f3d... is linked to node 89 (CLIPTextEncode)
# via link 214 which goes from the subgraph input (-10) to node 89
api_prompt['128']['inputs']['text'] = "Cinematic aerial shot of a misty mountain landscape at sunrise, golden light piercing through clouds, majestic peaks, smooth camera movement, photorealistic, ultra detailed, 8K"

# Set dimensions via width/height inputs on the grouped node
# The subgraph input maps width (id=bc890482...) to node 74 (EmptyHunyuanLatentVideo) via link 215
# But these are linked inputs, not widget overrides. For the grouped node to work,
# we need to provide them as inputs or the grouped node handles them internally.

# Actually, the proxyWidgets mechanism means the GROUPED node's widget values
# override the internal node's widget values. So setting widget values on node 128
# will propagate to the correct internal nodes.

# But the API format doesn't have "widgets_values" in the same way. For API prompts,
# we need to set the values directly on the internal nodes.

print(json.dumps(api_prompt, indent=2)[:2000])
print("---")
print("Submitting via API...")

r = requests.post(API, json={"prompt": api_prompt})
print(r.status_code, r.text[:1000])
