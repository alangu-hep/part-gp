import torch

def remove_all_forward_hooks( model: torch.nn.Module) -> None:

    print('\nRemoving all hooks within the model\n')
    
    for _, module in model.named_modules():
        module._forward_hooks.clear()

def remove_handles(handle_dict) -> None:

    print('\nRemoving all handles')
    
    for name, handle in handle_dict.items():
        handle.remove()

    print('\nHandles removed successfully\n')

def save_outputs(name, outputs):
    def hook(module, input, output):

        location = outputs[name]
        
        if isinstance(output, torch.Tensor):
            location.append(output.detach())
            return
        
        if isinstance(output, (tuple, list)):
            location.append(tuple(
                o.detach() if isinstance(o, torch.Tensor) else o
                for o in output
            ))
            return
            
        location.append(output)

    return hook


def register_forward_hooks(location_dict, outputs):
    """
    REQUIREMENTS:
    -location_dict must be in the format 'hook name': specific module within the model
    -outputs must be a dictionary
    """
    
    if not isinstance(location_dict, dict):
        print("location_dict must be in the format 'hook name': specific module within the model")
        return None

    for key in location_dict.keys():
        outputs[key] = []

    handles = {}
    
    for key, value in location_dict.items():
        
        hook_name = key

        try:
            handles[hook_name] = value.register_forward_hook(save_outputs(hook_name, outputs))
            print(f'\nForward Hook Registered: {hook_name}')
        except TypeError:
            print('Location Dict value must be a valid module for forward hook registration')
            _remove_all_forward_hooks(value)
            break

    return handles