import argparse

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import get_config
from swin_transformer import SwinTransformer


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer inference script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--ckpt-path', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--label-path', type=str, required=True, help='path to ImageNet-1K label file')

    # add image path to test
    parser.add_argument('--test-imgPath', type=str, required=True, help='path to single test image')
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def load_imgNet_1k_labels(label_path):
    with open(label_path, 'r') as label_file:
        labels_list = [line.rstrip('\n') for line in label_file]
    return labels_list

@torch.no_grad()
def model_infer(model, img_path, transform_pipe, top_k=3):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = transform_pipe(img).unsqueeze(0).cuda()

    output_logits = model(input_data)
    output_probs = F.softmax(output_logits, dim=1)
    _, output_idx = torch.topk(output_probs, top_k)
    output_idx = output_idx[0].tolist()

    return output_idx

def main(config):
    # ****************** build model ******************
    model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN.APE,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))
    if hasattr(model, 'flops'):
        flops = model.flops()
        print("number of GFLOPs: {}".format(flops / 1e9))

    # load pretrained model
    print("==============> Resuming form {}....................".format(config.MODEL.CKPT_PATH))
    checkpoint = torch.load(config.MODEL.CKPT_PATH, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print('model load result msg: {}'.format(msg))

    model.cuda()
    model.eval()

    # load imageNet-1K labels
    labels_list = load_imgNet_1k_labels(args.label_path)

    # data transforms
    transform_pipeline = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # input_data = torch.randn(1, 3, 224, 224).cuda()
    pred_idx_list = model_infer(model, args.test_imgPath, transform_pipeline, top_k=3)
    print('img_path: {}\npred_labels: {}'.format(args.test_imgPath, [labels_list[i] for i in pred_idx_list]))

if __name__ == '__main__':
    args, config = parse_option()
    main(config)
