# 独立 RVC 推理 API

从项目根目录运行，使用现有 `runtime`，不修改原始 RVC 源码。只支持推理，不提供训练接口。固定从 `rvc_models` 的一级模型文件夹发现权重和索引，支持中文名称。

## 启动与鉴权

```powershell
.\runtime\python.exe -m rvc_api --host 127.0.0.1 --port 8000
# 或运行 go-api-infer.bat
```

可选在启动前设置 `$env:API_BEARER_TOKEN = '你自己的令牌'`。配置后所有 `/api/v1/*` 要求 `Authorization: Bearer <令牌>`，同时关闭公开文档和 OpenAPI 路由；未配置则不验证。`GET /health` 始终公开，只返回 alive/ready，不返回模型、令牌或路径。令牌由环境提供，不写入仓库。

只运行一个服务进程、一个 worker；GPU 请求串行，占用时新请求返回 429。不要在同一 GPU 上启动多个 API 实例。HTTP 事件循环仍能处理健康检查等请求。

当前项目 runtime 已验证：Python 3.12.10、torch 2.7.1+cu128、FastAPI 0.99.1、Pydantic 1.10.26、Uvicorn 0.22.0，以及现有 FAISS、soxr、parselmouth、torchfcpe。还需要根目录 ffmpeg.exe/ffprobe.exe 和项目原有 HuBERT、RMVPE 辅助资源。API 复用当前环境，不需要升级项目依赖。

## 发现能力与模型选择

| 方法与路径 | 内容 |
|---|---|
| GET /api/v1/init | 模型、权重/索引候选、F0 工具可用性、参数范围、模式、传输协议、限额、显存策略 |
| GET /api/v1/models | 模型列表及扫描问题 |
| GET /api/v1/models/{model_id} | 单个模型及其参数能力 |
| GET /api/v1/f0-methods | pm、rmvpe、fcpe 的依赖及资源可用性 |
| GET /api/v1/modes | 当前支持的处理/索引模式 |
| POST /api/v1/infer | 原始音频流上传，multipart/mixed 音频和进度流返回 |

`model_id` 为文件夹名称，如 `芙宁娜`。`weight_id`、`index_id` 是发现接口返回的文件名称，不接受路径。只有一个候选时可以省略；多个候选必须明确选择。客户端对中文 query 参数进行 URL 编码。

Index 必须在所选模型文件夹内、能被 FAISS 读取、维度匹配 v1/v2，且向量数量满足检索需求。仅存在 `.index` 扩展名不足以启用。无可用 Index 时默认 rate 为 0，显式提交正数会返回错误；不会跨文件夹找索引。有可用 Index 时默认 rate 为 0.75。

发现接口读取 checkpoint 元信息到 CPU，不加载 GPU 模型；仅缓存基础元信息。只放入可信的本地模型 checkpoint。工具显示 available 表示依赖和资源检查通过，最终运行错误仍以推理响应为准。

## 请求与参数

请求的 Content-Type 为 `audio/*` 或 `application/octet-stream`，请求体是音频文件的原始字节，支持 HTTP chunked 上传。不要发送 multipart/form-data 或 JSON/base64 音频。

当前处理模式 `chunked_file`：流式接收上传，上传完成后解码、预处理和 F0 提取，再逐块推理、逐块发送。**不支持上传和推理同时进行的实时麦克风模式**。第一次加载模型和 F0 预处理有额外等待，音频返回后不再等待全部块推理完成。

| Query 字段 | 默认/限制 |
|---|---|
| model_id | 必填；模型文件夹名称 |
| weight_id / index_id | 可选；多候选时必须明确指定 |
| speaker_id | 0；0 到 speaker_count-1 |
| pitch_shift | 0；整数半音，仅 F0 模型支持 |
| f0_method | rmvpe；pm/rmvpe/fcpe |
| index_rate | 根据可用索引决定 0 或 0.75；范围 0～1 |
| index_mode | auto；off 强制关闭，required 要求可用索引且 rate>0 |
| resample_sr | 0 保留模型采样率；或 16000～48000 |
| rms_mix_rate | 0.25；0～1，1 不应用输入包络混合 |
| protect | 0.33；0～0.5，仅 F0 模型支持 |
| chunk_seconds | 5；1～30，按 16k 音频的 160 样本边界处理 |
| mode | chunked_file |

未知、重复、非有限值、越界参数被拒绝。非 F0 模型不执行 F0 提取，也不接受有效变调/保护调整。最终有效参数在 start/done 中返回。

参考客户端使用分块上传，增量解析返回内容，逐块写入 WAV，并打印当前/总共。收到成功 done 并检查完整性后才把 `.partial` 文件改名，失败会删除未完成文件。

```powershell
.\runtime\python.exe -m rvc_api.client input.wav output.wav --model 芙宁娜 --chunk-seconds 1 --f0-method rmvpe --resample-sr 24000
# 设置了鉴权时，客户端也从 API_BEARER_TOKEN 环境变量读取令牌
```

## 响应协议

HTTP 200 的 Content-Type 是 `multipart/mixed; boundary=rvc_<request_id>`。各 part 含 `Content-Length` 和 `X-Event-Type`；JSON 为 UTF-8，audio 为原始单声道 PCM16 little-endian。不是每块一个 WAV。使用长度解析二进制 part，不能在 PCM 数据中搜索 boundary 来切块。

事件顺序：`start -> audio -> progress -> ... -> audio -> progress -> done -> 关闭 boundary`。错误以 `error` 终止，不返回成功 done。一个 audio 后面的 progress 才表示该块已发送，current 从 1 到 total。

- start：request_id、codec=pcm_s16le、sample_rate、channels=1、total 和有效 parameters。
- audio：二进制音频，头部含 X-Chunk-Index（从 1 开始）、X-Sample-Offset、X-Sample-Count。拼接时验证偏移连续、字节数为 samples×2。
- progress：request_id、stage=infer、current、total、status=running。
- done：status=completed、current/total、samples、有效 parameters、gpu_cleanup_completed 及 CUDA allocated/reserved 测量值。
- error：status=failed、current/total、request_id、error.code/message 和清理结果。不要把此前音频当作完整成功结果。

首块之后的请求错误不能更改已经发送的 HTTP 状态码，必须读取 error/done 终态。初始化之前的错误是结构化 JSON，常见状态为 401 鉴权失败、413 超限、422 参数/音频/索引错误、429 忙碌、503 依赖或显存不足、408 超时。日志可按 X-Request-ID 定位。

代理应禁用响应缓冲；服务发送 X-Accel-Buffering: no，但代理最终行为需自行配置。客户端应按流读取响应；一次性读完响应无法体现首块延迟优势。

## 资源与限制

解码到 16k 单声道，至少 0.1 秒；最大上传默认 100 MiB，最大解码时长 600 秒。输入/输出包含非有限值时返回错误。相邻块使用上下文和交叉淡化，重采样保留流状态，PCM 输出限幅。

| 环境变量 | 默认 |
|---|---|
| API_MAX_REQUEST_BYTES | 104857600 |
| API_MAX_AUDIO_SECONDS | 600 |
| API_UPLOAD_TIMEOUT | 120 秒 |
| API_INFERENCE_TIMEOUT | 600 秒，包含准备及推理/背压等待 |
| API_DECODE_TIMEOUT | 60 秒 |

限额必须为正且有限。CPU 音频队列最多 2 块，同时保留最多 4 个响应；慢客户端产生背压。客户端断开时停止后续块，正在运行的 CUDA 操作完成后执行清理，不能强行中断 GPU kernel。

模型、HuBERT、F0 工具及 CUDA Graph 均为请求所有。最后一块计算完成后，在等待最后一块网络发送前清理 GPU；失败、超时、断连同样清理。临时文件和响应名额在请求结束时回收。

当前 runtime 的 CUDA Graph warmup 会保留 cuBLAS 工作区，API 在模型/图卸载后调用 torch 的 `_cuda_clearCublasWorkspaces`，再 empty_cache。该方法为 PyTorch 内部接口；升级 runtime 后必须重新跑显存验收。清理失败会标记 ready=false 并拒绝后续推理，应检查日志后重启。allocated/reserved=0 表示 PyTorch 的张量/缓存池已释放，CUDA 驱动上下文仍可能在任务管理器占用少量显存。

## 验证

项目根目录执行完整测试，含真实 CUDA、真实 TCP、中文路径、索引/无索引、三种 F0、重采样、首块提前发送、并发隔离、背压、断连、超时、实际 CUDA OOM、重复请求和原文件哈希校验：

```powershell
& .\runtime\python.exe -m unittest discover -s tests_rvc_api -p "test_*.py" -v 2>&1 | Tee-Object -FilePath "build.log"
$buildExitCode = $LASTEXITCODE
if ($buildExitCode -ne 0) { Get-Content build.log -Tail 120; exit $buildExitCode }
```

测试按本项目的现有模型配置运行。测试音频为生成的合成信号，用于确认推理和协议行为；不代表主观歌声/音色质量验收。生成 WAV 和测量 JSON 位于 `tests_rvc_api/artifacts`，不提交模型或测试音频。阶段记录见 `docs/fastapi实施进度.md`。
