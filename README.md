## 特性 Features

* 路径追踪全局光照 Path tracing Global-illumination
![](./screenshot/cornell_box.png)

* 多重重要性采样 Multiple importance sampling
![](./screenshot/veach_mis.png)

* hdri重要性采样 Hdri importance sampling
![](./screenshot/hdri_test.png)

* 微表面分布 Microfacet distribution
![](./screenshot/material_test.png)

* 迪士尼原型材质 disney principled material
![](./screenshot/disney_hyperion.png)

* 薄镜模型景深 depth of field

* 【材质】金属、玻璃、塑料、迪士尼 [material] metal glass plastic disney

* 【灯光】定向光、环境、矩形、球、圆盘、聚光灯 [light] directional dome rect point disk spot

## 截图 screenshot

![](./screenshot/dining_room.png)

![](./screenshot/staircase.png)

![](./screenshot/staircase2.png)

![](./screenshot/mustang_fastback.png)

## build
Cmake + Visual Studio + Optix7.6.0

## todo
* volume absorption + volume scatter
* mipmap + ray cone
* sobol sampler
* temporal denoise