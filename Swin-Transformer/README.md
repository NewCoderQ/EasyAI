# Swin-Transformer


## Inference
```python
CUDA_VISIBLE_DEVICES=0 python3 main.py \
                            --eval \
                            --label-path test_data/ImageNet-1K_labels \
                            --cfg swin_small_patch4_window7_224.yaml \
                            --ckpt-path ckpts/swin_small_patch4_window7_224.pth \
                            --test-imgPath test_data/dog-puppy-on-garden.jpg
```