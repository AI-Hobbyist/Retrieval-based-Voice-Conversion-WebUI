# 推理模型目录

每个一级子目录为一个 `model_id`，放置直接包含的 RVC 推理 `.pth` 及可选 `.index`。文件夹内有多个权重/可用索引时需要显式指定 ID。无可用索引时 Index Rate 默认 0；禁止从其他文件夹借用索引。

本地验收将现有 assets/weights 中四个权重复制至同名模型文件夹，前三个复制同名 assets/indices 索引；原文件保留。模型二进制不提交 Git。新增模型请使用可信来源，checkpoint 会经 torch.load 读取。
