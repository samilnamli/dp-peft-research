import random
import numpy as np
import torch
import os
from typing import Dict, Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_environment_info() -> Dict[str, Any]:
    env_info = {
        'python_version': os.sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        env_info['device_name'] = torch.cuda.get_device_name(0)
        env_info['device_capability'] = torch.cuda.get_device_capability(0)
    
    try:
        import transformers
        env_info['transformers_version'] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import opacus
        env_info['opacus_version'] = opacus.__version__
    except ImportError:
        pass
    
    try:
        import peft
        env_info['peft_version'] = peft.__version__
    except ImportError:
        pass
    
    return env_info
