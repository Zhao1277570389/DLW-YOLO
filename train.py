from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8SwinTransformer.yaml")

#    ultralytics/cfg/models/v8/yolov8.yaml
#    ultralytics/cfg/models/v8/yolov8-AKConv.yaml
#    ultralytics/cfg/models/v8/yolov8-DSConv.yaml
#    ultralytics/cfg/models/v8/yolov8-ODConv.yaml
#    ultralytics/cfg/models/v8/yolov8CondConv.yaml
#    ultralytics/cfg/models/v8/yolov8DWConv.yaml
#    ultralytics/cfg/models/v8/yolov8Ghostconv.yaml
#    ultralytics/cfg/models/v8/yolov8LightConv.yaml
#    ultralytics/cfg/models/v8/yolov8RepConv.yaml
#    ultralytics/cfg/models/v8/yolov8-C2f-LSKA.yaml
#    ultralytics/cfg/models/v8/yolov8-CrissCrossAttention.yaml
#    ultralytics/cfg/models/v8/yolov8-DSConv+C2f_DySnakeConv.yaml
#    ultralytics/cfg/models/v8/yolov8-EfficientNetv2.yaml
#    ultralytics/cfg/models/v8/yolov8-EMA.yaml
#    ultralytics/cfg/models/v8/yolov8-Glod.yaml
#    ultralytics/cfg/models/v8/yolov8-pose.yaml
#    ultralytics/cfg/models/v8/yolov8-RepViTblock.yaml
#    ultralytics/cfg/models/v8/yolov8-SOCA.yaml
#    ultralytics/cfg/models/v8/yolov8-vanillanet.yaml
#    ultralytics/cfg/models/v8/yolov8BiFormer.yaml
#    ultralytics/cfg/models/v8/yolov8bifpn.yaml
#    ultralytics/cfg/models/v8/yolov8BoTNet.yaml
#    ultralytics/cfg/models/v8/yolov8CARAFE.yaml
#    ultralytics/cfg/models/v8/yolov8CBAM.yaml
#    ultralytics/cfg/models/v8/yolov8ContextAggregation.yaml
#    ultralytics/cfg/models/v8/yolov8HorNet-Backbone.yaml
#    ultralytics/cfg/models/v8/yolov8HorNet-head.yaml
#    ultralytics/cfg/models/v8/yolov8Involution.yaml
#    ultralytics/cfg/models/v8/yolov8jct.yaml
#    ultralytics/cfg/models/v8/yolov8MobileOne.yaml
#    ultralytics/cfg/models/v8/yolov8MobileViT.yaml
#    ultralytics/cfg/models/v8/yolov8RepLKNet.yaml
#    ultralytics/cfg/models/v8/yolov8SEAttention.yaml
#    ultralytics/cfg/models/v8/yolov8ShuffleNetV2.yaml
#    ultralytics/cfg/models/v8/yolov8slimneck.yaml
#    ultralytics/cfg/models/v8/yolov8slimneck+GSConv.yaml
#    ultralytics/cfg/models/v8/yolov8SwinTransformer.yaml

results = model.train(data="/root/zz/zz.yaml", epochs=100, imgsz=640)

#     /root/bf/bf.yaml
#     /root/cm/cm.yaml
#     /root/hs/hs.yaml
#     /root/xm/xm.yaml
#     /root/yc/yc.yaml
#     /root/yx/yx.yaml
#     /root/zz/zz.yaml
