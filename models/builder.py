import torch
from .expalignet import expalignet
from .vltrdetrnet import vlrtdetrnet

def build_model(args):
    """
    统一的模型构建工厂 (Factory)。
    支持动态构建不同的多模态检测架构。
    
    Args:
        args (dict | Namespace): 包含模型超参数的配置对象。
                                 至少需要包含 'model_name' 字段。
    Returns:
        nn.Module: 构建好并推送到对应 device 的模型实例。
    """
    # 如果 args 是 argparse.Namespace，将其转换为 dict 方便统一处理
    if not isinstance(args, dict):
        args = vars(args)
        
    # 1. 提取通用配置
    model_name = args.get('model_name', 'vlrtdetrnet').lower()
    text_embed_dim = args.get('text_embed_dim', 256)
    size = args.get('size', 'tiny')
    num_classes = args.get('num_classes', 80)
    device = args.get('device', 'cuda')
    pretrained = args.get('pretrained', None)

    # 2. 路由分发 (根据名字造模型)
    if model_name == 'expalignet':
        # 提取 expalignet 专属配置
        reg_max = args.get('reg_max', 16)
        
        model = expalignet(
            text_embed_dim=text_embed_dim,
            size=size,
            pretrained=pretrained,
            num_classes=num_classes,
            reg_max=reg_max
        )
        # print(f"[expalignet - {size}]")

    elif model_name == 'vlrtdetrnet':
        model = vlrtdetrnet(
            text_embed_dim=text_embed_dim,
            size=size,
            pretrained=pretrained,
            num_classes=num_classes
        )
        # print(f"[vlrtdetrnet - {size}]")

    else:
        raise ValueError(f"不支持构建模型: {model_name}。请检查配置！")

    # 3. 统一处理 Device 挂载
    model.to(device)
    
    # 注意：expalignet 的 DetectionLoss 在初始化时绑定了 parameters().device
    # 但模型在 __init__ 时默认在 CPU 上。
    # 为了防止 Loss 里的张量报错，我们在 to(device) 后强制更新一下 loss_fn 的 device
    if hasattr(model, 'loss_fn') and hasattr(model.loss_fn, 'device'):
        model.loss_fn.device = torch.device(device)

    return model