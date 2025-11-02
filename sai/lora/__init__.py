import torch


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}

        has_lora_A = any(k.endswith(".lora_A.weight") for k in lora_state_dict)
        has_lora_down = any(k.endswith(".lora_down.weight") for k in lora_state_dict)

        if has_lora_A:
            lora_a_keys = [k for k in lora_state_dict if k.endswith(".lora_A.weight")]
            for lora_a_key in lora_a_keys:
                base_name = lora_a_key.replace(".lora_A.weight", "")
                lora_b_key = base_name + ".lora_B.weight"

                if lora_b_key in lora_state_dict:
                    target_name = base_name.replace("diffusion_model.", "", 1)
                    lora_name_dict[target_name] = (lora_b_key, lora_a_key)

        elif has_lora_down:
            lora_down_keys = [
                k for k in lora_state_dict if k.endswith(".lora_down.weight")
            ]
            for lora_down_key in lora_down_keys:
                base_name = lora_down_key.replace(".lora_down.weight", "")
                lora_up_key = base_name + ".lora_up.weight"

                if lora_up_key in lora_state_dict:
                    target_name = base_name.replace("lora_unet_", "").replace("_", ".")
                    target_name = target_name.replace(".attn.", "_attn.")
                    lora_name_dict[target_name] = (lora_up_key, lora_down_key)

        else:
            print(
                "Warning: No recognizable LoRA key names found in state_dict (neither 'lora_A' nor 'lora_down')."
            )

        return lora_name_dict

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        lora_name_dict = self.get_name_dict(state_dict_lora)
        updated_num = 0

        lora_target_names = set(lora_name_dict.keys())
        model_layer_names = {
            name for name, module in model.named_modules() if hasattr(module, "weight")
        }
        matched_names = lora_target_names.intersection(model_layer_names)
        unmatched_lora_names = lora_target_names - model_layer_names

        print(f"Successfully matched {len(matched_names)} layers.")
        if unmatched_lora_names:
            print(
                f"Warning: {len(unmatched_lora_names)} LoRA layers not matched and will be ignored."
            )

        for name, module in model.named_modules():
            if name in matched_names:
                lora_b_key, lora_a_key = lora_name_dict[name]
                weight_up = state_dict_lora[lora_b_key].to(
                    device=self.device, dtype=self.torch_dtype
                )
                weight_down = state_dict_lora[lora_a_key].to(
                    device=self.device, dtype=self.torch_dtype
                )

                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(
                        2
                    ).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)

                if module.weight.shape != weight_lora.shape:
                    print(f"Error: Shape mismatch for layer '{name}'! Skipping update.")
                    continue

                module.weight.data = (
                    module.weight.data.to(weight_lora.device, dtype=weight_lora.dtype)
                    + weight_lora
                )
                updated_num += 1

        print(f"LoRA loading complete, updated {updated_num} tensors in total.\n")
