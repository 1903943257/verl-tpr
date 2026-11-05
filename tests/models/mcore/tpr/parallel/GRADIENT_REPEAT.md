# Base CP1 首次反向重复检查

两个独立 torchrun 进程依次使用同一张 NPU，各自加载相同的原始 checkpoint，
重放历史实验的 11 次前向预热和 step-0 probe，使用原始第一批 PPO 数据反向一次。
历史 `actor_fixture.pt`、old_log_probs、advantages 和结果目录只读。
沿用 1K prompt + 1K response、两条轨迹、microbatch=1、BF16 网络、FP32 main_grad、
原来的梯度累加融合设置，以及真实 VERL/Megatron/MindSpeed 训练路径。

在原生 finalize 完成后，调用原生优化器 `prepare_grads()` 和两次 `get_grad_norm()`，
要求 BF16 loss scale=1、无 grad scaler，随后读取全部优化器拥有的 FP32 main_grad。
不执行梯度裁剪、Adam 更新、scheduler step，也不替换 attention、linear、finalize 或 norm 算子。
优化器必须提供这些原生方法，缺失时直接报错，不使用自制范数替代原生结果。
CPU NumPy FP64 分块统计仅作旁路对照，覆盖全部参数元素。

宿主机拉取当前分支后，在之前运行实验的容器内执行：

```bash
export TPR_TRAIN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_TRAIN10_SOURCE=/workspace/tpr_experiments/p1024s1024_train10_1
export TPR_GRAD_REPEAT_DEVICE=0  # 与原始 Base CP1 使用同一张物理卡
export TPR_GRAD_REPEAT_OUTPUT=/workspace/tpr_experiments/p1024s1024_grad_repeat_1
bash tests/models/mcore/tpr/parallel/run_training_gradient_repeat.sh
cat "$TPR_GRAD_REPEAT_OUTPUT/summary.md"
```

输出目录必须是新的，并与历史目录分开。无须容器内 Git 可用。
`comparison.json` 包含全元素梯度 rel L2/cosine/max_abs、CPU FP64 范数、
每次运行的两次原生范数、两组初始 probe/训练 logprob 差异，以及 loss 差异。
`second/parameter_differences.json` 记录每个参数的误差统计，不保存逐元素梯度值。
`runtime_audit_rank0.json` 记录原生 schedule、attention、finalize、融合 wgrad 和 norm 方法来源。

符号统计区分严格正负翻转和零/非零转换，同时给出全部元素和双方非零元素的分母。
`TPR_GRAD_REPEAT_SMALL_THRESHOLD` 默认 `1e-6`，以未裁剪、未更新的梯度绝对值为单位，
记录参考梯度绝对值不超过阈值的数量及其中的翻转/零转换数量，不把这些计数当成训练精度验收阈值。

磁盘临时保存第一组的一份 FP32 梯度；第二组分块读取比较，不写第二份。
1.7B 模型约需 6.4 GiB 临时空间，预检额外要求 1 GiB 余量。
比较成功后默认只清理本次 manifest 指定的梯度块，保留 JSON、前向小张量、日志及索引。
失败时保留临时梯度以便检查；设置 `TPR_GRAD_REPEAT_KEEP_GRADS=1` 可在成功后也保留。
显存不包含 Adam 更新后才创建的矩状态；实际峰值和分阶段耗时记录在各组 `result.json`，
总时间包含两次模型加载、前向预热以及约 6.4 GiB 梯度的写入和读取。

若两次完整训练 logprob 相同而 main_grad 不同，分歧已在反向/累加/同步阶段体现；
这仍不证明所有前向中间量相同。若 main_grad 完全相同而原生 norm 不同，则优先检查范数计算。
若梯度和 norm 都相同，本次未复现历史漂移，不能据此宣称历史漂移不存在。
CPU/原生范数有小差异本身不证明出错，还需考虑统计精度和归约顺序。
这两次检查不会建立稳定的噪声分布，也不会单独定位到某个反向算子。
