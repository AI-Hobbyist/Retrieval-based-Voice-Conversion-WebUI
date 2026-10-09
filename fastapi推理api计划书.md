# FastAPI 推理 API 计划书

日期：2026-10-09。本文为当前项目的静态分析和后续实现计划；本次只新增此文档，不修改、运行或修复原始代码，不表示 API 已实现或性能已验证。所有“API 默认值”“校验规则”“路由”均为拟定契约；“当前源码”描述已有行为。

## 1. 目标和范围

在现有 RVC 离线推理核心上新增独立 FastAPI 服务：客户端上传音频，选择 `rvc_models` 下的模型文件夹，提交推理参数，获得转换后的音频。模型根目录固定为项目根目录下的 `rvc_models`，不得通过请求更换根目录。

模型以文件夹为单位展示与隔离；只有所选模型文件夹包含可用 `.index`，才能启用 Index Rate。没有索引的模型仍可执行普通推理，但 Index Rate 必须为 0。

首版提供 init 初始化、模型/F0 工具/模式发现、参数能力查询、音频分块流上传与返回、可选 Bearer Token 校验和健康检查。训练、特征提取、创建索引、模型上传、模型编辑、实时麦克风、声卡控制、WebSocket 音频流、人声分离及 PyMSS 不进入本计划的 API 范围。批量文件可由客户端依次调用单文件接口，不另增批量任务系统。

首版先收完上传音频并完成解码/预处理，然后按固定计划逐块推理：每个音频块完成后立即返回该块及当前/总共进度，不等待整段推理完成。上传块、HTTP 传输块和推理音频块是不同边界；本次不要求边上传边推理。返回为带音频和进度事件的 `multipart/mixed` 数据流，客户端解析后可逐块播放或拼接 PCM 并写 WAV。

## 2. 当前源码结构及调用路径

| 职责 | 现有实现 | 证据 |
|---|---|---|
| 离线界面和默认参数 | Gradio 模型推理页 | `webui.py:1422–1505` |
| 加载 RVC 权重和元信息 | `VC.get_vc`：读取 checkpoint、识别版本/F0/采样率/说话人数，创建合成器和 Pipeline | `infer/vc/modules.py:115–149` |
| 单文件转换入口 | `VC.vc_single` | `infer/vc/modules.py:164–247` |
| 音频解码 | `load_audio`：默认单声道；CUDA torchaudio 路径及 FFmpeg 回退 | `infer/audio.py:167–178,310–335` |
| 离线转换流水线 | `Pipeline.pipeline`，索引、滤波、分段、F0、特征及合成、包络和重采样 | `infer/vc/pipeline.py:255–410` |
| 文件输出格式 | `VC.vc_multi` 可写 WAV/FLAC，其他格式通过 `wav2` 转码 | `infer/vc/modules.py:308–343` |
| 实时转换 | 独立 `infer/rtrvc.py` 与 `realtime_gui.py` 路径 | `infer/rtrvc.py:135–146,225` |

离线调用顺序：

```text
加载选定 .pth → 创建对应 v1/v2、F0/非 F0 合成器和 Pipeline
上传文件 → 解码成 16 kHz 单声道 → 峰值归一化（必要时）
→ 加载本地 HuBERT → 可选 FAISS 特征检索
→ 高通滤波及长音频分段 → 可选 F0 提取和半音变调
→ HuBERT 特征 / 检索特征融合 → 可选清辅音保护 → RVC 合成
→ 音量包络混合 → 可选最终重采样 → int16 单声道输出
```

输入归一化、HuBERT 按需加载和调用证据为 `infer/vc/modules.py:180–217`。当前 Pipeline 虽内部分段（`infer/vc/pipeline.py:286–307`），但会先 concatenate 全部合成片段，再执行整段 RMS、重采样和全局输出峰值归一化，最后一次性返回 int16（同文件 `:395–410`）；它没有逐块返回音频/进度接口。因此只给完整结果套 StreamingResponse 不能满足需求。新 API 必须增加专用分块适配器，复用 `Pipeline.get_f0` / `Pipeline.vc`，原核心文件不修改。

## 3. 仅推理可调项

### 3.1 客户端每次推理可调参数

`VC.vc_single` 本身没有这些参数的 Python 默认值；下表“现有默认”来自 WebUI。API 进行显式校验，不能把 WebUI 范围误当作核心已经执行的校验。

| API 字段 / 核心参数 | 现有默认和范围 | 拟定 API 默认和约束 | 实际作用、条件和源码证据 |
|---|---|---|---|
| `model_id` | UI 选择 `.pth` 名称，无固定默认 | 必填；模型文件夹 ID | 选择音色/权重。API 改为以文件夹组织；核心加载方式见 `infer/vc/modules.py:115–147` |
| `weight_id` | UI 每项对应一个权重 | 文件夹恰有一个 `.pth` 时可省略；多个时必填 | 仅允许当前文件夹枚举出的权重，不接受路径 |
| `speaker_id` / `sid` | 默认 0；UI 加载模型后动态最大值 | 默认 0；整数 `0 <= speaker_id < n_spk` | 选择模型内部说话人，不等同于模型文件夹 ID。`n_spk` 源于 `emb_g.weight`；`infer/vc/modules.py:123,148`、`infer/vc/pipeline.py:308`。不能直接沿用 UI 的 `maximum=n_spk`，嵌入下标上界应为 `n_spk-1` |
| `pitch_shift` / `f0_up_key` | 0，整数半音；UI 未定义上下界 | 默认 0；严格整数，不截断小数；首版不凭空增加音质范围 | +12 升八度，-12 降八度，按 `2^(key/12)` 缩放 F0。`webui.py:1448–1451`、`infer/vc/modules.py:178`、`infer/vc/pipeline.py:130` |
| `f0_method` | `rmvpe`；`pm/rmvpe/fcpe` | 默认 `rmvpe`；仅三个枚举值；返回运行环境中的可用性 | F0 提取算法；`infer/vc/pipeline.py:64–123`，UI `webui.py:1453–1457`。F0 固定基频边界 50–1100 Hz 是内部参数，非请求字段 |
| `index_id` / `file_index` | WebUI 可填任意索引路径 | 只接受所选文件夹内的索引 ID；恰有一个可用索引时可省略；多个且 rate>0 时必填 | API 禁止任意索引路径；rate=0 时不加载索引。`infer/vc/pipeline.py:273–285` |
| `index_rate` | 0.75；0–1 | 有可用 index：默认 0.75；无可用 index：默认 0；范围 `[0,1]` | 检索特征占比：0 跳过检索；1 全用检索特征。融合公式为 `检索特征*rate + 原特征*(1-rate)`；`webui.py:1494–1500`、`infer/vc/pipeline.py:177–196` |
| `resample_sr` | 0；UI 0–48000 | 默认 0；仅接受 0 或整数 16000–48000 | 0 保留模型目标采样率；其他值仅当不同于模型采样率且 >=16000 才生效。拒绝 UI 虽可输入、核心却忽略的 1–15999，防止误导。`webui.py:1467–1474`、`infer/vc/pipeline.py:398–401` |
| `rms_mix_rate` | 0.25；0–1 | 默认 0.25；有限实数 `[0,1]` | 越接近 0 越匹配输入音量包络，越接近 1 越保留合成输出包络；1 跳过包络调整。`webui.py:1475–1483`、`infer/vc/pipeline.py:23–42,395–397` |
| `protect` | 0.33；0–0.5；UI 步长 0.01 | 默认 0.33；有限实数 `[0,0.5]`，不强制 UI 步长 | 清辅音/呼吸保护；0.5 关闭，降低值增强保护且可能减弱索引效果；需要 F0。`webui.py:1484–1493`、`infer/vc/pipeline.py:199–216` |
| `chunk_seconds` | 当前核心没有该 API 参数 | API 新设计，默认 5 秒；有限实数 1–30 秒，最终取整到 16 kHz/160 样本帧边界 | 每个交付音频块的主内容长度；总块数在推理前固定，最后不足一块计一个；范围是拟定服务契约，不是原源码范围 |
| `output_format` | 批量输出支持格式转换 | 音频 part 固定 `pcm_s16le` 单声道，整体响应 `multipart/mixed`，非可调字段 | 每块返回无 WAV 头的 PCM16；客户端按 start 的采样率与声道拼接/写 WAV；JSON 进度不混入音频字节，FLAC/MP3 不属于首版 |

对于 `if_f0=0` 权重，F0 算法、半音变调和清辅音保护不生效（`infer/vc/pipeline.py:309–320,199–216`）。能力响应标记 `supports_f0=false`；省略相关字段时允许使用默认值并标记未应用；明确提交非默认变调、算法或保护设置时返回 422，避免表面成功却未应用。数值参数必须拒绝 NaN/Infinity。过大半音引起溢出时应返回可识别错误；首版若要增加产品级半音上下限，需在实施前明确契约，不以“源码范围”名义杜撰。

### 3.2 服务级、内部推理配置

以下会影响推理速度/资源，但不作为普通请求参数或允许客户端随时切换：

| 项目 | 当前行为 | 首版处理 |
|---|---|---|
| 设备和 FP16/FP32 | `Config` 使用 `infer_device/infer_dtype`；根据硬件分支配置，`configs/config.py:157–180,222–286` | 启动时确定，受保护 init 可报告；不提供每请求设备/精度切换 |
| 原长音频分段 `x_pad/x_query/x_center/x_max` | FP16：3/10/60/65；FP32：1/6/38/41；显存 <=4GB：1/5/30/32（秒）；`configs/config.py:253–270` | 保留原核心事实；API 用 chunk_seconds 另建交付计划，上下文参考 x_pad，不把原切点当交付块 |
| CUDA Graph | 当前可由 `RVC_CUDA_GRAPH` 与 `RVC_CUDA_GRAPH_MAX_CACHE` 管理 | 服务启动级设置，复用现状；模型切换时正确清理缓存，不承诺实际加速收益 |
| 固定推理常量 | 16 kHz 特征输入、160 样本帧；48 Hz 高通；FAISS 查询 k=8；F0/阈值常量，`infer/vc/pipeline.py:20,54–61,74–75,106,122,186` | 记录实现事实，不扩展 API 参数，不引入调优/重构任务 |

实时链路额外具有 `formant_shift`（共振峰变化）及设备、音频块和延迟等设置。`infer/rtrvc.py:138–146` 的 formant/index 热切换不能据此认定离线 `VC.vc_single` 支持共振峰；首版离线请求不接受 `formant_shift`。训练轮次、batch size、学习率、训练采样率、GPU 训练分配等均不属于推理参数。

## 4. 固定模型目录与模型发现契约

### 4.1 目录格式

```text
项目根目录/
  rvc_models/
    音色A/
      voice.pth
      added_voice.index       # 可选；没有时禁用 Index Rate
    音色B/
      voice.pth               # 只有权重也能推理
    多权重音色/
      version1.pth
      version2.pth
      feature_a.index
      feature_b.index
```

`model_id` 对应 `rvc_models` 的一级子文件夹；仅枚举各文件夹直接包含的 `.pth` / `.index` 普通文件，首版不递归合并嵌套目录。扩展名匹配不区分大小写。中文/空格名称可显示，客户端使用服务返回的 ID；大小写冲突和 ID 冲突在发现时报告，不能随机选择。

不搬动原有权重或生成索引；目录为空/不存在时返回空列表和明确状态，推理返回模型不存在。实施时由管理员将可信模型放入此固定目录。

本次目录盘点：`rvc_models` 尚不存在；现有 `assets/weights` 含 `guanguanV1.pth`、`keruanV1.pth`、`kikiV1.pth`、`youzhanv2-xi.pth`，`assets/indices` 中前三者有同名 `.index`，`logs` 中也有重复索引；`indices` 有 `anchun_added_IVF2695_Flat_nprobe_1_anchun_v2.index`，不能把它自动配给 `youzhanv2-xi`。将来整理目录可把前三组分别放入带 index 的模型文件夹，`youzhanv2-xi` 单独作为无 index 文件夹；这仅为未来整理建议，本次不搬迁。文件名不能证明实际 version/F0/采样率，必须加载可信 checkpoint 验证。

### 4.2 索引规则

1. 所选文件夹没有 `.index`：`has_index=false`、`index_rate_enabled=false`、默认 rate=0。显式 rate>0 或传 `index_id` 返回 422 `INDEX_NOT_AVAILABLE`；绝不跨文件夹寻找 index。
2. 有 `.index` 但损坏、无可重建向量、维度不匹配：保留 `has_index=true`，能力中标明不可用及原因；有可用索引时才 `index_rate_enabled=true`。rate>0 请求在推理前失败，不能静默降级为无索引成功。
3. 单个可用 index 可自动选；多个可用 index 且 rate>0 时要求显式 `index_id`，没有元数据不按文件名猜测 `.pth` 与 `.index` 的训练配对。维度相同仅证明结构兼容，不证明音色来源正确。
4. 选中索引需可由 FAISS 读取并 `reconstruct_n`；校验特征维度：v1 为 256、v2 为 768；由于当前查询 k=8，首版可将少于 8 个向量的索引判为不适用于此检索路径。证据：`infer/vc/pipeline.py:186,279–280`；HuBERT 版本维度见 `infer/hubert.py:68–127`。
5. `index_rate=0` 不调用 FAISS，也不要求选择 index；返回 `index_used=false`。rate>0 必须记录实际使用的 `index_id` 和 rate。
6. 权重和索引在加载/执行前重新确认文件存在、边界和指纹；禁止软链接/junction/reparse point 逃出固定根目录，不依赖字符串前缀判定安全边界。

原 `get_vc` 自动匹配索引使用全局目录策略，不能作为新目录契约的来源；适配层仅使用当前模型文件夹的索引。`VC.vc_single` 还会执行 `.replace("trained","added")`，作用于完整路径，可能改写目录名（`infer/vc/modules.py:189–199`）。新增适配器直接加载已验证的原始索引绝对路径及向量，并传入 `Pipeline.vc`，避开全局搜索、路径改写和索引静默失败入口。原文件不因此修改。

## 5. 独立服务设计

拟新增一个独立 API 包及入口文件，职责分别为请求/响应 schema、模型目录注册表、离线推理适配器、生命周期/路由；本次仅描述，不创建这些代码。

适配器只导入 `infer` 内必要模块，不导入 `webui.py` 启动 Gradio；与原应用分开运行。加载权重采用与 `VC.get_vc` 一致的 checkpoint 识别/合成器构造流程，但从固定目录注册表取得绝对路径，避免靠每次请求改 `weight_root` 等进程环境变量。

`Config` 是单例包装，初始化会 `arg_parse()`/`parse_args()`（`configs/config.py:157–180,190–209`），直接接在 Uvicorn CLI 后可能误解析服务参数。独立适配层构造仅包含 VC/Pipeline 所需字段的推理配置，复用现有设备选择结果/分段策略；不能通过临时改全局 `sys.argv` 绕过，也不修原 `Config`。

基础依赖是当前兼容的 Python/PyTorch/FAISS/音频依赖，加 FastAPI 与 Uvicorn；上传直接接收音频字节流，无需 multipart 表单解析依赖；返回使用 multipart/mixed 协议封装。HuBERT 使用本地 Transformers 目录 `assets/hubert_base`，以 `local_files_only=True` 加载（`infer/hubert.py:22,31–57`）；RMVPE 使用 `rmvpe_root/rmvpe.pt`（`infer/vc/pipeline.py:95–106`）；FCPE 按现有 `FCPEInfer` 依赖实际检查。服务不自动下载、训练或替换辅助模型。

生命周期初始化注册表和推理工作线程；init/模型能力元数据仅 CPU 读取 checkpoint 并释放引用，不在初始化时常驻 GPU 模型。推理请求内按需加载权重/辅助模型，完成自动释放；关闭时等待活动推理收尾并清理临时文件。FastAPI 生命周期机制依据 [Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)。

单进程、单 worker，首版每次仅允许一个实际推理操作；锁覆盖选权重、加载/切换模型、推理与缓存更新，避免两个请求互相换模型。推理放到专用单线程执行器，异步路由保持事件循环可响应健康请求；限制在途请求和等待时间，超容量立即返回 429 `INFERENCE_BUSY`，不建立无限队列。多 worker 会复制模型内存，相关依据为 [Deployment Concepts](https://fastapi.tiangolo.com/deployment/concepts/)；同步/异步执行依据为 [Concurrency and async / await](https://fastapi.tiangolo.com/async/)。

模型与缓存仅在同一个推理请求的各块之间复用；不保留跨请求 RVC/HuBERT/F0 模型 GPU 缓存。当前 Pipeline 会在每次有索引推理时读 FAISS 并重建向量，新适配器只在本请求内复用已验证索引。请求超时/断开不能当作计算已停止：线程/GPU 仍运行时继续持有推理锁，实际结束并清理后才能释放；不强行终止线程、不一边执行一边卸载模型。

### 5.1 新增逐块推理适配器（未来实现）

上传完成后执行全段解码、16 kHz 单声道转换、原输入归一化和高通滤波；F0 模型复用 `get_f0` 完成整段 F0/半音预处理，非 F0 跳过。随后以 chunk_seconds 和 160 样本帧对齐建立固定主内容区间，`total=ceil(有效输入帧数/每块帧数)`，最后短块计一个。预处理可能耗时，不能把上传/预处理耗时说成第 1 块推理进度。

每块只取主内容加左右上下文（参考当前 x_pad），F0 按同一时间轴切片；`Pipeline.vc` 执行 HuBERT/可选 FAISS/保护/合成。裁掉上下文后，以带上下文的局部 RMS 调整实现原 rms_mix_rate 语义；跨块保留少量 RMS 插值状态，避免独立片段包络跳变。必要的重采样使用有状态转换器，保留滤波历史/相位并在末尾 flush；不对每个片段无状态重复重采样。按累计输出采样点规划边界，保证采样偏移连续、没有重复或遗漏。

边界处理采用有界上下文、短 holdback 和 crossfade：仅保留尚未提交的短尾部，在下一块边界完成混合；首块稳定的主内容立即发送，不等待全部块。协议每块对应连续、不重叠的最终交付 PCM 区间，待混合尾部归入下一交付块；最后块包含全部残留/重采样 flush，仍保持固定 total。holdback 长度是服务内部有界配置，应在 start 中报告；不能为消除接缝把整个输出留到最后。

原全局输出峰值归一化需要看到未来全部输出，不能在已发送块上追改增益。API 明确改为逐块 limiter/防削波：每块最终 PCM 前检查有限值和峰值，限制幅度并采用连续平滑增益/有界前瞻，限制器状态跨块保留；不得直接 int16 强转溢出。此局部 RMS、接缝和输出限幅策略属于新增 API 适配，与整段原 Pipeline 输出可能不同，不承诺比特一致；音质/接缝验收通过才可交付。

推理锁覆盖整个请求 GPU 会话，包括所有块、状态更新和安全收尾；生产者逐块计算，音频队列最多 2 块，消费者发送一块后继续，慢客户端通过背压限制后续生产，禁止提前计算并堆积全部输出。断开/超时停止调度新块，已启动的 GPU 工作完成后才能释放锁。新增适配器在当前请求一次性加载/验证索引并复用至所有块，检索失败上抛，不静默关闭 Index Rate。

### 5.2 推理完成后自动释放显存

完成最后一块合成及重采样 flush 后，将最终音频/进度所需数据转为 CPU PCM，等待本请求 GPU 工作同步完成，立即执行 owner 级 cleanup；不等待 HTTP 下载全部结束，也不要求客户端调用卸载接口。所有正常、异常、超时/断连路径进入同一 finally：清除本请求 RVC/HuBERT/RMVPE/FCPE 与合成中间张量、GPU 重采样器等引用，按 owner 清 CUDA Graph 缓存，执行 `gc.collect()` 和 `torch.cuda.empty_cache()`，然后释放推理会话锁。CPU 音频队列/网络缓冲在响应结束时单独回收。

最后块转 CPU 后，应先完成 GPU cleanup，再执行可能受网络背压阻塞的最后块/终态入队操作，避免慢客户端使已结束推理的模型继续驻留。队列、生成器闭包和异常对象不得保留 GPU 张量/模型引用；cleanup 失败时不能发送 `gpu_cleanup_completed=true` 的成功终态，应记录失败并将推理服务标记为未就绪，阻止未清理会话与后续请求混用。

不得在仍有 worker/GPU kernel 使用模型时抢先删除，取消请求需停止新块调度并等待已启动工作结束后 cleanup。不用每块 empty_cache；请求内复用保证不为每块重载模型。所有仍在 GPU 的共享/全局引用都应纳入所有权清单：`infer.audio` 有 GPU resampler 缓存，API 首版宜独立 CPU 解码/16 kHz 输入重采样，避免创建难以归属的全局 GPU 音频缓存；输出有状态重采样也用 CPU 状态，其他 GPU 缓存若被复用，必须有不修改原文件的 owner 级释放办法才能通过验收。

done 在推理 cleanup 完成后才入队，携带 `gpu_cleanup_completed=true`；error 在可安全收尾后同样标记清理结果，不以进程重启作为卸载策略。CUDA context/第三方持有内存可能仍存在，不承诺 nvidia-smi 显存绝对为 0；验收应观察 PyTorch allocated/reserved 与 nvidia-smi 进程使用量回到稳定空闲基线、多次请求无累积增长，二者指标不等价。

## 6. API 契约（拟定）

| 方法 / 路由 | 用途 | 返回 |
|---|---|---|
| `GET /health` | 公开最小存活探测 | 200，仅 alive/ready；不返回模型、设备、绝对路径或详细异常 |
| `GET /api/v1/init` | 客户端 init 节点获取初始化数据 | models、按权重能力、f0_methods、modes、参数默认/范围和 index 条件、上传/输出媒体协议、auth_required、请求限制；鉴权开启时需要 Token |
| `GET /api/v1/models` | 扫描固定目录并按文件夹列出模型 | 模型 ID/名称、权重/索引 ID、是否包含 index、是否可用、问题列表 |
| `GET /api/v1/models/{model_id}` | 返回该模型的权重能力/推理参数 | 按 weight_id 的版本、F0、n_spk、目标采样率、索引候选、默认/范围；元信息仅 CPU 按需读取，不常驻显存 |
| `GET /api/v1/f0-methods` | F0 提取工具选择 | pm/rmvpe/fcpe 的名称、可用性、不可用原因、默认项；“代码支持”与“依赖/权重可用”分别表示，不谎报已加载验证 |
| `GET /api/v1/modes` | 模式选择 | 返回 `processing_modes`、`index_modes` 及各项参数要求；处理模式首版为 chunked_file；索引模式为 auto/off/required，均映射到已有检索开关 |
| `POST /api/v1/infer` | 上传文件，逐块推理并立即返回音频和进度 | 200 `multipart/mixed; boundary=...`，StreamingResponse 依次发送 start/audio/progress/done 或 error part；开始前失败返回结构化 JSON |

`POST /api/v1/infer` 的 body 是音频原始字节，`Content-Type` 为支持的 `audio/*` 或 `application/octet-stream`；参数使用 query（第 3 节字段及 `mode=chunked_file`），不混入 JSON Body，不整包调用 `request.body()`。通过 `async for chunk in request.stream()` 边接收边校验累计大小、写服务私有临时文件；完成后验证解码与时长再推理。官方依据为 [Request.stream](https://fastapi.tiangolo.com/reference/request/)。HTTP/1.1 chunked 与 HTTP/2 DATA 块均按服务收到的字节串顺序写入，不要求固定上传块长度。

返回用 `StreamingResponse` 消费逐块生产队列，媒体类型为 `multipart/mixed`。每个完成的推理音频块立即生成二进制 audio part，接着生成 progress JSON；不可先生成完整 WAV 再读文件假装逐块推理。part 内可按 64 KiB 传输，HTTP/TCP 会拆分或合并数据，客户端以 MIME boundary 和长度解析 part，不能把一次网络 read 当一次推理块。官方依据为 [StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/) 和 [RFC 2046 §5.1/5.1.3 multipart](https://www.rfc-editor.org/info/rfc2046/)。模型选择以每请求 `model_id/weight_id` 为准，不设置全局“选中模型”；动态 index_rate 默认在选中模型后解析。

### 6.1 音频与进度 part 协议

每个 part 有 `Content-Type`、`Content-Length` 与 `X-Event-Type`。JSON 使用 application/json；音频使用 application/octet-stream，codec 在 start 明确为 `pcm_s16le`（不能用表示大端 PCM 的 audio/L16）。事件顺序：

1. `start`：request_id、mode、codec、sample_rate、channels=1、总块数 total、current=0、有效参数/index 信息、holdback 和状态 running。此时 chunk plan 已固定。
2. `audio`：实际二进制 PCM，带 `X-Chunk-Index`（从 1 起）、`X-Sample-Offset`（最终输出起始采样点，从 0 起）、`X-Sample-Count`、Content-Length；满足 length=sample_count*channels*2。
3. `progress`：每发送完一个完整 audio part，发送 `{stage:"infer",current:i,total:N,status:"running"}`。current 表示已完成并交给 HTTP 发送的完整音频块数，客户端收到 progress 后确认已接收对应 audio；不等同于远端已播放。开始 0/N，最后 N/N。
4. 成功 `done`：current=N、total=N、status=completed、总采样点数、最终有效参数及 gpu_cleanup_completed=true，然后关闭 MIME boundary。只有所有块成功且推理显存 cleanup 完成才发送；音频块在各自完成后已即时交付。
5. 中途失败 `error`：code/message、status=failed、current=已发送块数、total=N，随后关闭 boundary；不得补发成功 done。若连接已断则无法发 error，客户端在缺失 done/完整终止符时必须视为未完成。

首版响应开始前完成上传/预处理；这些阶段在服务日志/init 契约中使用 stage=upload/preprocess、current/total=null，不伪造推理进度。开始响应前的错误仍可用第 7 节 HTTP 错误码；响应头发出后不能改状态码，失败通过 error part 表达。客户端不得把 HTTP 200 本身视为完整成功。

响应结构示意（省略部分 MIME 头；二进制标记是说明文字，非实际 payload）：

```text
Content-Type: multipart/mixed; boundary=rvc_REQUEST_ID

--rvc_REQUEST_ID
Content-Type: application/json
X-Event-Type: start

{"request_id":"...","codec":"pcm_s16le","sample_rate":40000,"channels":1,"current":0,"total":3,"status":"running"}
--rvc_REQUEST_ID
Content-Type: application/octet-stream
Content-Length: 400000
X-Event-Type: audio
X-Chunk-Index: 1
X-Sample-Offset: 0
X-Sample-Count: 200000

<第 1 块 PCM 二进制>
--rvc_REQUEST_ID
Content-Type: application/json
X-Event-Type: progress

{"stage":"infer","current":1,"total":3,"status":"running"}
... 后续 audio / progress ...
--rvc_REQUEST_ID
Content-Type: application/json
X-Event-Type: done

{"current":3,"total":3,"status":"completed","samples":600000,"gpu_cleanup_completed":true}
--rvc_REQUEST_ID--
```

客户端从 start 确定格式，增量解析 MIME，按 chunk index/offset 验证连续性，仅把 audio part 写入 PCM 缓冲/播放器；progress 更新 `当前/总共`。保存 WAV 时先写占位头并顺序追加 PCM，在成功 done 后按实际采样点数回填 RIFF/data 长度；失败保留为不完整结果或删除，不把 JSON/边界字节写入 WAV。无需新增 WebSocket 或任务查询系统。

模型能力示例（结构示意，不是当前目录实测）：

模式选择也通过每次推理的 query 明确提交，不改变其他请求的默认选择：

- `mode=chunked_file`：上传完成后逐块推理并即时返回音频/进度；不接受未实现的实时麦克风模式。
- `index_mode=auto`（默认）：按所选权重的可用索引解析默认 rate；无索引且未指定 rate 时为 0，有可用索引时为 0.75。显式非零 rate 仍必须有可用索引。
- `index_mode=off`：禁止检索，省略 rate 时取 0；显式非零 rate 或指定 index_id 视为冲突并返回 422。
- `index_mode=required`：要求选中可用索引且有效 rate>0；省略 rate 时取 0.75，无可用索引或显式 rate=0 返回 422。多个候选仍须 index_id。

`f0_method` 是 F0 工具选择，F0/非 F0 是权重固有能力，不能通过模式参数把非 F0 权重变成 F0 模型。`GET /api/v1/init` 中的模式数据采用相同契约，例如：

```json
{
  "auth_required": true,
  "models": [],
  "models_state": "directory_missing",
  "f0_methods": [
    {"id": "pm", "available": true, "verification": "dependency_check"},
    {"id": "rmvpe", "available": false, "reason": "辅助权重未就绪"},
    {"id": "fcpe", "available": false, "reason": "依赖未就绪"}
  ],
  "modes": {"processing_modes": ["chunked_file"], "index_modes": ["auto", "off", "required"]},
  "audio_transport": {"upload": "raw_body_stream", "download": "multipart_mixed_audio_progress", "codec": "pcm_s16le", "inference_starts": "after_upload_and_preprocess", "audio_emitted": "after_each_inference_chunk", "progress": "current/total"}
}
```

以上仅示意响应结构，F0 可用性值必须来自实际环境检查；完整 init 还包含第 3 节参数 schema、服务限制和能力验证状态。浏览客户端在取得模型/权重能力后才能决定显示哪些调节项。

```json
{
  "model_id": "voice_a",
  "display_name": "音色 A",
  "has_index": true,
  "weights": [{"weight_id": "voice.pth", "version": "v2", "supports_f0": true,
    "speaker_count": 1, "sample_rate": 40000,
    "index_rate_enabled": true, "default_index_rate": 0.75}],
  "indexes": [{"index_id": "added_voice.index", "usable": true, "dimension": 768}],
  "issues": []
}
```

请求示例（计划实现后才可执行）：

```powershell
curl.exe --http1.1 -X POST `
  "http://127.0.0.1:8000/api/v1/infer?model_id=voice_a&mode=chunked_file&pitch_shift=0&f0_method=rmvpe&index_rate=0.75&resample_sr=0&rms_mix_rate=0.25&protect=0.33" `
  -H "Content-Type: audio/wav" `
  -H "Transfer-Encoding: chunked" `
  -H "Authorization: Bearer YOUR_TOKEN" `
  --data-binary "@D:/audio/input.wav" `
  --output "D:/audio/response.multipart.bin"
```

此 curl 命令只保存整个 multipart 响应，得到的 `.bin` 不是 WAV，必须按上述协议解析 audio part 后组装 WAV；真实使用客户端应增量解析，以边接收音频边显示 progress。无索引模型省略 index_rate 或传 0。服务文件名由服务生成，输入多声道混为单声道，首版不承诺保留立体声。

### 6.2 可选 Bearer Token 验证

服务启动配置 `API_BEARER_TOKEN`：未配置/为空时关闭鉴权；非空时 `/api/v1/init`、模型/F0/模式能力接口和推理接口均要求 `Authorization: Bearer <token>`。`/health` 始终只公开最小存活/就绪状态；开启鉴权时 `/docs`、`/redoc` 和 OpenAPI schema 禁用，避免出现无保护业务信息入口。

使用 `HTTPBearer(auto_error=False)` 解析可选凭据，由应用根据服务配置决定是否必须提供，并进行常量时间 Token 比较；缺失、格式错误或不匹配均返回 401 和 `WWW-Authenticate: Bearer`。鉴权必须在读取上传流和加载模型前执行，不把 token 放 query、响应或日志。Token 仅通过进程环境/受保护配置注入，不写入源码。依据 [FastAPI Security Reference](https://fastapi.tiangolo.com/reference/security/)；HTTPBearer 只解析凭据，真实 Token 校验仍由应用实现。远程使用 Bearer 时通过 HTTPS 部署。

## 7. 文件、错误和资源边界

每个请求使用服务临时根目录下的随机私有目录；流式分块存储上传并计数，不以 MIME/后缀作为唯一音频有效性依据。禁止服务器任意文件路径、URL、目录输入、客户端输出路径与命令字符串；探测/解码仅针对保存的本地文件，以参数数组调用工具。仅管理员放入的可信 checkpoint 可被加载，因为现有流程使用 `torch.load`（`infer/vc/modules.py:121`）。

首版服务默认上传上限建议 100 MiB、解码时长上限 600 秒，同时限制解码后的样本数；这些是拟定资源策略，不是原源码限制或硬件性能结论。具体部署值由服务配置设置，不开放给推理请求。仅检查压缩字节不足以限制解码内存，必须在完整推理前验证音频时长、样本数量、非空及有限值，并为解码设置超时/资源边界。极短音频可能无法满足 `filtfilt`/reflect padding（`infer/vc/pipeline.py:286–307`）；预检应给可解释的 `AUDIO_TOO_SHORT`，不修改生产滤波行为。

| 状态码 | 错误 code | 触发条件 |
|---|---|---|
| 401 | `UNAUTHORIZED` | 配置要求 Token 时凭据缺失/格式错误/不匹配 |
| 404 | `MODEL_NOT_FOUND` / `WEIGHT_NOT_FOUND` | ID 不在注册表中 |
| 413 | `AUDIO_LIMIT_EXCEEDED` | 文件字节数/时长/解码样本超限 |
| 422 | `INVALID_PARAMETER` / `INDEX_NOT_AVAILABLE` / `AMBIGUOUS_SELECTION` | 数值/枚举无效、无 index 却启用、多个权重/索引未选择 |
| 422 | `INVALID_AUDIO` / `AUDIO_TOO_SHORT` / `INCOMPATIBLE_INDEX` | 解码失败、空/非有限输入、太短、索引损坏或维度不匹配 |
| 429 | `INFERENCE_BUSY` | 在途推理容量已满 |
| 503 | `DEPENDENCY_UNAVAILABLE` / `MODEL_LOAD_FAILED` / `GPU_OUT_OF_MEMORY` | 辅助资源缺失、权重加载失败、显存耗尽 |
| 500 | `INFERENCE_FAILED` | 未分类的实际推理/输出错误 |

响应开始前错误统一为 `{"request_id":"...","error":{"code":"...","message":"..."}}`；开始后使用第 6.1 节 error part 和 failed 终态。原始 traceback/绝对路径只进入服务日志。每块发送前检查采样率、有限值、PCM 长度和连续偏移；成功要求全部 audio/progress 及 done 完整，不以 HTTP 200 或中文状态字符串判定成功。

索引读取错误目前被 Pipeline 捕获后继续无索引推理（`infer/vc/pipeline.py:278–283`）；API 需在执行前完成 FAISS 验证，并使用请求期间不会变化的服务持有文件快照策略。预检和快照仍不能排除执行时内存不足等失败，因此适配层必须有能确认实际索引加载与检索的执行路径：在新增 API 包内实现最小索引适配，索引异常向上抛出，其他推理步骤保持原逻辑；不得仅原样调用会吞掉索引异常的入口后宣称已用 index。若直接复用原 Pipeline 无法提供该保证，此项为实现阻塞，必须先完成适配再通过验收。不能单凭文件存在或 `vc_single` 的索引文本断言“实际已检索”；原核心文件和异常策略保持不变。

默认只监听 `127.0.0.1`；外网暴露需要部署层访问控制和上传限制。日志记录 request_id、model_id/weight_id/index_id、有效参数、设备、耗时、错误分类，不保存音频正文。成功文件在响应发送完成后清理，失败也清理；进程异常留下的临时目录按启动时约定回收，禁止对计算出的任意路径进行递归删除。

## 8. 实施阶段和验收

以下是未来实施顺序，不在本次文档任务中执行。每阶段只做所列范围；自动验收通过即完成，发现训练/GUI/其他模块问题记后续事项。若实施时有可用 Git 仓库，按项目规则更新进度、检查 status/diff、提交该阶段并 push 且确认成功后才进入下一阶段；当前目录没有可依赖的提交基线，不为文档任务擅自创建仓库/远端。

### 阶段 1：模型注册表和参数契约

新增固定目录发现、文件夹/权重/索引 ID、schema 与能力元信息读取；完成 init、模型/F0/模式选择契约、多候选选择规则、路径边界、动态 Index Rate 默认值与可选 Token 规则；只新增 API 所需文件。

验收：无索引默认 rate=0，正值拒绝；有可用索引默认 0.75；坏 index 不可用；跨目录 ID/路径、遍历和链接逃逸拒绝；单/多权重与索引行为确定；speaker 上界正确；参数拒绝小数半音/NaN/无效重采样。

### 阶段 2：离线适配器和 HTTP 接口

实现独立启动配置、lifespan、单线程逐块适配器（复用 get_f0/vc）、固定 chunk plan、局部后处理/有状态重采样、Request.stream 上传、multipart StreamingResponse 音频/进度、Bearer 校验、自动释放显存/错误清理及上述路由；保持原程序和源代码不变。

验收：可信 F0/非 F0、有/无 index 场景逐块返回音频，客户端能组装可解码 WAV；采样率/连续偏移/总样本数正确，最后短块和重采样 flush 不丢失或重复；变调/RMS/保护/重采样参数可追溯，非 F0 不默默应用 F0 参数。用跨块发声/静音/峰值样本检查接缝、平滑增益和无 int16 削波溢出；明确与原整段输出的差异，不强制比特一致。原 Gradio 不受影响。

### 阶段 3：资源与串行隔离验证、交付

补齐范围内必要的并发、容量、输入上限、模型切换、断开/异常清理验证，并提供启动和请求说明；不扩展队列后台任务、实时或分离功能。

验收：至少 N>=3 的输入记录第 1 块客户端收到时间与第 N 块推理完成时间，证明前者更早；每块到达后立即有相应 progress，0/N→N/N 单调且总数固定；原始 HTTP 块任意拆/并仍能正确解析 part。慢消费者验证队列最多 2 块和背压，不预先完成整段推理；服务无完整输出缓存。中途显存/索引/合成失败产生 error/failed，绝无成功 done；断连/超限/超时正确收尾并在实际 GPU 完成前保持锁。最后块推理结束后自动 cleanup，即使下载仍未结束显存也回稳定空闲基线；done 在 cleanup 后发送。重复请求记录 allocated/reserved 与 nvidia-smi，模型/图/重采样缓存无累积引用或持续增长，异常/断连同样释放，init 不常驻 GPU 模型。Token、init、模型/模式/F0 能力和文件夹 index 限制符合契约；并发 A/B 不串模型，繁忙 429，health 保持响应。原代码无改动。记录设备、块数、首块等待/每块耗时/总耗时，不预设性能承诺。

所有未来安装、构建、测试命令必须按 AGENTS 使用前台 PowerShell，`2>&1 | Tee-Object -FilePath "build.log"`，立即检查 `$LASTEXITCODE`，失败阅读日志。本计划仅针对 API 不要求 Qt 验证；若后续验收涉及 Qt GUI，只允许真实 Windows 窗口，禁止 offscreen；无法运行则记 `MANUAL/PENDING`，不能作为 GUI PASS。

## 9. 已知限制与证据边界

- 本次没有执行模型加载、音频推理、性能测试或 GUI 验证；文中例子不代表目录已创建、模型可加载或 index 匹配已证实。
- 核心的索引静默失败、路径字符串替换、Gradio 返回结构和 Config 参数解析是新适配层必须隔离的现状，不在本次修复。
- 清辅音保护依赖 pitchf；当前 F0 代码会尝试插值原来为零的 F0（`infer/vc/pipeline.py:125–139`），实际保护效果要通过代表性样本验证，不能只从 UI 标签保证效果。
- 极短/全静音、FAISS 零距离和边界数据的数值问题可能影响输出；在实施时纳入当前请求有效性/错误处理所需验证，不因此重构训练或合成器。确需原代码修改或范围扩张时另行取得明确授权。
- 官方 FastAPI 文档只支撑框架使用方法；模型目录、参数和索引规则来自用户要求及当前源码。上述 API 契约和实施方案均是本项目设计建议。
