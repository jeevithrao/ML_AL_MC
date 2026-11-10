import timm

def make_model(name="efficientnet_b0", n_classes=182, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, in_chans=1,
                             num_classes=n_classes, drop_rate=0.3, drop_path_rate=0.1)
