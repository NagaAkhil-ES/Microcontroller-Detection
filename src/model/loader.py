from model.frcnn_lit import FasterRCNNLightning

def get_model(model_arch, num_classes, lr):
    """API to get model of type nn.module

    Args:
        model_arch (string): model architecture
        num_classes (int): number of classes (background inclusive)
        lr (float): learning rate

    Returns:
        model.frcnn_lit.FasterRCNNLightning: FRCNN lighting model
    """
    if model_arch == "frcnn-mobilenet":
        model = FasterRCNNLightning(backbone="mobilenet", 
                                    num_classes=num_classes, lr=lr)
        
    return model

# testing block
if __name__ == "__main__":
    import pdb; pdb.set_trace()
    model = get_model("frcnn-mobilenet", 2, 1e-3)