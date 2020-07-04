from .detr import DETR


def build_detr_resnet50(num_classes=91, num_queries=100):
    from .backbone import ResNet50Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet50Backbone(name='backbone/0/body'))


def build_detr_resnet50_dc5(num_classes=91, num_queries=100):
    from .backbone import ResNet50Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet50Backbone(
                    replace_stride_with_dilation=[False, False, True],
                    name='backbone/0/body'))


def build_detr_resnet101(num_classes=91, num_queries=100):
    from .backbone import ResNet101Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet101Backbone(name='backbone/0/body'))


def build_detr_resnet101_dc5(num_classes=91, num_queries=100):
    from .backbone import ResNet101Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet101Backbone(
                    replace_stride_with_dilation=[False, False, True],
                    name='backbone/0/body'))
