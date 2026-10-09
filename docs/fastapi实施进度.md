# 独立 FastAPI 实施进度

约束：仅新增独立 API、测试和相关文档；原始 RVC 代码保持不变。Python、测试和推理都使用项目 `runtime/python.exe`。分支 `api-infer`，远端由用户指定。

当前项目的原文件与远端 main 存在预先差异；只暂存本次新增文件，不提交这些原文件差异。阶段严格依次推进，验证通过后更新本记录、检查 staged diff、提交并成功 push，再开始下一阶段。

| 阶段 | 内容 | 状态 | 验证 |
|---|---|---|---|
| 1 | 固定文件夹注册表、CPU 元信息、参数和索引模式契约 | 已提交并推送 `6fd16ee` | runtime unittest：8/8 PASS，无跳过 |
| 2 | 逐块推理、HTTP 音频/进度流、鉴权和显存清理 | 已提交并推送 `c908a97` | runtime unittest：22/22 PASS，含实际 CUDA 推理 |
| 3 | 并发/背压/断连/显存及实际推理验证、使用说明 | 自动验收通过；本记录随阶段检查点提交推送 | runtime 全套 31/31 PASS，补充客户端截断回归 1/1 PASS，无跳过 |

阶段 1 验证命令：`runtime/python.exe -m unittest discover -s tests_rvc_api -p test_registry.py -v`（前台 PowerShell Tee build.log）。覆盖无/有效/坏/少向量/维度错误索引、跨文件夹禁止、多候选显式选择、索引模式、非 F0、说话人上界、中文路径、Windows junction 逃逸、参数边界。FAISS 中文路径通过 API 内 Python IO 回调读取，无原代码修改。checkpoint 元信息仅 CPU 读取，缓存仅保留基础类型。

阶段 2：`runtime/python.exe -m unittest discover -s tests_rvc_api -p "test_*.py" -v`，22 项通过，耗时 33.220 秒。实际设备 NVIDIA GeForce RTX 5060 Laptop GPU，Python 3.12.10 / torch 2.7.1+cu128 / FastAPI 0.99.1 / Pydantic 1.10.26。

- 真实权重：guanguanV1 + index + RMVPE + 变调/24k；youzhanv2-xi 无 index + FCPE/48k；用户新增中文路径 芙宁娜/芙宁娜.pth + index + PM/32k。每个 3.17 秒输入交付 4 块，采样数和偏移正确。
- 非 F0 使用测试时生成的有效小型随机 checkpoint 验证真实合成路径，不将其宣称为训练完成的音色模型；测试后删除该临时模型文件夹。
- 完成后各场景 allocated/reserved=0；曾发现 CUDA Graph warmup 流的 cuBLAS 工作区缓存残留，API 在原始模型/图卸载后使用当前 PyTorch 提供的 `_cuda_clearCublasWorkspaces` 回收。新增针对该残留的回归测试，原文件保持不变。
- HTTP 测试覆盖 init/模型/F0/模式、鉴权开启/关闭、PCM/progress/done 顺序、索引条件、结构化失败终态、输入上限、清理失败标记未就绪、原文件哈希；测试中的故障日志是预期注入，不是未解决错误。
- 阶段 2 推送确认后进入阶段 3，以下使用真实 TCP，不以 TestClient 缓冲响应代替证据。

阶段 3：全套 `runtime/python.exe -m unittest discover -s tests_rvc_api -p "test_*.py" -v`，31 项通过，耗时 58.575 秒；随后新增 `test_client.py` 的截断响应回归，1 项通过。全部前台 PowerShell Tee build.log，立即检查退出码。

- 真实 Uvicorn/TCP + 原始 chunked 上传：芙宁娜 → guanguanV1 → 芙宁娜，RMVPE + index + 24k，每次 6.17 秒输入输出 7 块、148080 样本，进度严格 1～7，偏移连续。首块收到分别在提交后约 4.657/3.141/3.485 秒，早于最后一块完成约 1.609/1.422/1.281 秒。首次冷加载在独立测试进程中会更慢，不将这些时间作为延迟保证。
- 每次完成的 GPU allocated/reserved 都是 0；init 不增加 CUDA 张量。真实 CUDA OOM 在第 2 块发生时交付第 1 块后返回失败终态，未返回 done，随后显存/临时目录释放。真实推理断连停止后续块并清理模型。
- 并发模型 A/B 的独立网络验证：运行时 B 返回 429，健康检查正常；完成后 B 可运行，音频内容不混用。慢客户端将推理阻塞在少量块，队列峰值为 2；TCP reset 后清理在 3 秒内完成。另验证上传大小、上传超时/中断、推理超时以及失败不保存完整 WAV。
- 背压断连曾暴露生成器挂起于 ASGI send 时 finally 不及时执行的问题。新增响应所有者在 ASGI 请求退出时主动清理并关闭迭代器，生成器与响应共用幂等清理函数；真实网络回归通过。
- 原始 ffmpeg/ffprobe 解码验证中文输入文件名、立体声、时长上限、过短和无效音频。参考客户端收到 done 但缺少结束 boundary 时仍删除 `.partial`，避免误保存不完整响应。
- 使用说明见 `rvc_api/README.md`，启动器为 `go-api-infer.bat`。模型二进制、测试 WAV/JSON、runtime 不纳入提交。原始文件 SHA256 全部保持不变。
- 合成测试音频验证运行和协议行为，主观歌声/音色质量为 MANUAL/PENDING，不阻塞自动验收检查点。没有 Qt GUI 改动或 Qt offscreen 测试。
