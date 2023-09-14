import re

import torch
from PIL import Image
from torchvision.transforms import transforms

from transfer_net import TransformerNet


# 加载图片
def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


# 保存图片
def save_image(filename, data):
    img = data.cpu().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


# 图片处理
def stylize(input_addr, output_addr, style_path):
    content_image = load_image(input_addr)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(style_path)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        output = style_model(content_image.cuda()).cuda()
        save_image(output_addr, output[0])
