
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Creates a ResNet18 model for binary image classification.

    Classes:
        0 = Real image
        1 = AI-generated image

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: Whether to freeze the feature extractor.

    Returns:
        Configured ResNet18 model.
    """

    # Load pretrained weights if requested
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)

    # Freeze pretrained feature extraction layers
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Replace final classification layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features=num_features,
        out_features=2
    )

    return model


if __name__ == "__main__":

    # Create model without downloading pretrained weights
    model = create_resnet18(pretrained=False)

    # Create dummy images: batch of 4 RGB images
    dummy_images = torch.randn(4, 3, 224, 224)

    # Run forward pass
    model.eval()

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Model: ResNet18")
    print("Output shape:", outputs.shape)
    print("Predicted classes:", outputs.argmax(dim=1))
