# only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    from .nodes import NODE_CLASS_MAPPINGS

    NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

    # Monkey-patch LTXAVTEModel.load_sd to load connector weights alongside Gemma3 GGUF
    import comfy.text_encoders.lt as lt_mod

    _orig_load_sd = lt_mod.LTXAVTEModel.load_sd

    def _patched_load_sd(self, sd):
        has_gemma = "model.layers.47.self_attn.q_norm.weight" in sd
        has_connector = "text_embedding_projection.video_aggregate_embed.weight" in sd

        if has_gemma:
            result = self.gemma3_12b.load_sd(sd)
            if has_connector:
                proj_prefix = "text_embedding_projection."
                vw = sd.get(proj_prefix + "video_aggregate_embed.weight")
                target_w = self.text_embedding_projection.weight
                if (
                    vw is not None
                    and vw.ndim == 2
                    and vw.shape[0] >= target_w.shape[0]
                    and vw.shape[1] == target_w.shape[1]
                ):
                    self.text_embedding_projection.weight.data.copy_(
                        vw[: target_w.shape[0]].to(
                            device=target_w.device,
                            dtype=target_w.dtype,
                        )
                    )
                vb = sd.get(proj_prefix + "video_aggregate_embed.bias")
                target_b = self.text_embedding_projection.bias
                if vb is not None and target_b is not None:
                    if vb.ndim == 1 and vb.shape[0] >= target_b.shape[0]:
                        self.text_embedding_projection.bias.data.copy_(
                            vb[: target_b.shape[0]].to(
                                device=target_b.device,
                                dtype=target_b.dtype,
                            )
                        )
            return result
        else:
            return _orig_load_sd(self, sd)

    lt_mod.LTXAVTEModel.load_sd = _patched_load_sd
