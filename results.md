## Performance Comparison on PlantVillage and PlantDoc Datasets

| Model | Accuracy (PV) | Macro-F1 (PV) | Accuracy (PD) | Macro-F1 (PD) | Accuracy Drop | Macro-F1 Drop |
|------|---------------|---------------|---------------|---------------|---------------|---------------|
| DenseNet121 | 0.9975 | 0.9960 | 0.7800 | 0.7500 | 0.2175 | 0.2460 |
| ResNet50 | 0.9968 | 0.9956 | 0.7800 | 0.7400 | 0.2168 | 0.2556 |
| VGG16 | 0.9957 | 0.9943 | 0.5700 | 0.4900 | 0.4257 | 0.5043 |
| **MobilePlantViT** | **0.9957** | **0.9945** | **0.7900** | **0.7500** | **0.2057** | **0.2445** |
| ShuffleNetV2 | 0.9912 | 0.9855 | 0.7700 | 0.7200 | 0.2212 | 0.2655 |
| MobileNetV3-Small | 0.9806 | 0.9800 | 0.7100 | 0.7100 | 0.2706 | 0.2700 |
| SqueezeNet | 0.9800 | 0.9762 | 0.5300 | 0.4400 | 0.4500 | 0.5362 |

**Note:**  
- PV: PlantVillage dataset  
- PD: PlantDoc dataset  
- Accuracy Drop = Accuracy(PV) − Accuracy(PD)  
- Macro-F1 Drop = Macro-F1(PV) − Macro-F1(PD)

/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage/Tomato_Septoria_leaf_spot/0ab271a7-765e-4675-8bfc-e249c0c86fdd___Keller.St_CG 1778.JPG