# 独立 FastAPI 实施进度

约束：仅新增独立 API、测试和相关文档；原始 RVC 代码保持不变。Python、测试和推理都使用项目 `runtime/python.exe`。分支 `api-infer`，远端由用户指定。

当前项目的原文件与远端 main 存在预先差异；只暂存本次新增文件，不提交这些原文件差异。阶段严格依次推进，验证通过后更新本记录、检查 staged diff、提交并成功 push，再开始下一阶段。

| 阶段 | 内容 | 状态 | 验证 |
|---|---|---|---|
| 1 | 固定文件夹注册表、CPU 元信息、参数和索引模式契约 | 自动验收通过，准备提交推送 | runtime unittest：8/8 PASS，无跳过 |
| 2 | 逐块推理、HTTP 音频/进度流、鉴权和显存清理 | 未开始 | 未执行 |
| 3 | 并发/背压/断连/显存及实际推理验证、使用说明 | 未开始 | 未执行 |

阶段 1 验证命令：`runtime/python.exe -m unittest discover -s tests_rvc_api -p test_registry.py -v`（前台 PowerShell Tee build.log）。覆盖无/有效/坏/少向量/维度错误索引、跨文件夹禁止、多候选显式选择、索引模式、非 F0、说话人上界、中文路径、Windows junction 逃逸、参数边界。FAISS 中文路径通过 API 内 Python IO 回调读取，无原代码修改。checkpoint 元信息仅 CPU 读取，缓存仅保留基础类型。
