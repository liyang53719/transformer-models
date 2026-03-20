# RMSNorm SRAM 流水化调试总结

## 1. 背景

这轮调试围绕 `rmsNormalization` 的 SRAM 化流水线展开，目标有两个：

1. 保持 DDR 读、SRAM 写、SRAM 读和输出乘法的重叠，提高整体利用率。
2. 修正流水线边界处的数值和时序问题，确保波形行为与控制逻辑一致。

本轮最终落地了两类修复：

1. `invRms` 输出侧增加独立锁存，修正 token 尾拍的数值错位。
2. 新 token 启动加载时同拍发起首个 DDR 读，去掉 token 边界的 `ddrReadEn` 气泡。

## 2. 现象与定位

### 2.1 数值问题

早期 SRAM 双 bank 方案虽然已经实现了真实的读写重叠，但在 token 尾部存在数值错误。问题表现为：

1. 大部分输出正确。
2. 仅在 token 最后若干拍出现误差。
3. 误差与 `invRms` 在输出阶段跨 token 切换时机有关。

根因是输出路径直接使用“当前选中的 `invRms`”，而不是对“本 token 输出真正应该使用的 `invRms`”做单独保持。于是当下一个 token 的 `invRms` 提前到达时，尾拍会被新 token 的缩放因子污染。

### 2.2 波形问题

用户进一步指出 `tb_rmsNormalization_fsdb.dut.CoreDUT_ddrReadEn` 在约 `3905000` 附近出现下降沿，不符合预期。

针对这一点，先从生成 RTL 和 testbench 的层级入手，确认：

1. testbench 时基是 `1ns / 1ps`。
2. 时钟为 `always #5 clk = ~clk`，完整周期 10ns。
3. 问题信号位于 `dut.u_CoreDUT.u_Controller` 控制路径下。

加入定点调试打印后，抓到了关键序列：

1. `ddrReadAddr=190`
2. `ddrReadAddr=191`
3. 下一拍 `requestBeatIndex=192`，但 `loadActive` 仍为 1，`ddrReadEn=0`
4. 再下一拍才发出新 token 的 `ddrReadAddr=192`

这证明问题不是加载状态提前撤销，而是 token 边界多出了一拍空泡。

## 3. 根因分析

### 3.1 `invRms` 尾拍错位

在原结构里：

1. `InvRmsLatch0/1` 已经分别保存了每个 bank 的 `invRms`。
2. 输出阶段通过 bank 选择器直接取当前 `SelectedInvRms`。
3. 但输出路径没有再对“输出 token 的 `invRms`”做一次锁存。

因此，bank 选择条件一旦切到下一个 token，尾拍就可能读到错误的 `invRms`。

### 3.2 `ddrReadEn` token 边界气泡

控制器模板原先的执行顺序是：

1. 若 `loadActive` 为真，则调用 `iIssueRead(...)` 发起当前 token 的读请求。
2. 当当前 token 的 `requestBeatIndex` 达到 `beatsPerToken` 后，这条路径不再发读。
3. 只有在稍后的 `canStartLoad` 分支里，才把下一个 token 的 `loadActive` 拉起并把计数器清零。
4. 但该分支没有在同一个周期立即调用 `iIssueRead(...)`。

结果就是：

1. 旧 token 最后一拍读完。
2. 边界处出现 1 拍 `ddrReadEn=0`。
3. 新 token 的首拍 DDR 读延后一拍发出。

这个行为数值上未必立刻出错，但它确实违反了预期时序，并且直接损伤利用率。

## 4. 修复方案

### 4.1 输出侧新增 `OutputInvRmsLatch`

在 `buildRmsNormalizationModel.m` 中重构了流式 DUT：

1. 将累加、`rsqrt`、`invRms` 锁存改为 bank 化结构。
2. 分离 `BeatAccumulator0/1`、`ScalarRsqrt0/1`、`InvRmsLatch0/1`。
3. 新增 `OutputInvRmsLatch`，在输出 token 起始时对被选中的 `invRms` 再锁存一次。

这样输出阶段整 token 使用的是稳定的 `invRms`，不再受下一 token 到达时机影响。

### 4.2 控制器模板去掉边界气泡

在 `transformer_simulink/layer/private/rmsNormalizationControllerTemplate.txt` 的 `canStartLoad` 分支中，新增了一次立即发起首拍读请求的调用：

1. 新 token 被允许启动加载。
2. 设置 `loadActive`、`loadTokenIndex`、bank 状态。
3. 复位 `requestBeatIndex` 和 `receiveBeatIndex`。
4. 同拍调用 `iIssueRead(...)` 发出新 token 的第一拍 DDR 读取。

核心原则是：

1. 控制器一旦决定开始下一个 token 的 load，首拍读请求必须同拍产生。
2. 不能把“开始加载”和“发起第一拍请求”拆成两个周期。

## 5. 验证结果

### 5.1 定点窗口验证

修复前，目标窗口中读地址序列为：

1. `190`
2. `191`
3. 空泡
4. `192`

修复后，目标窗口变为：

1. `190`
2. `191`
3. `192`
4. `193`

说明 token 边界的 `ddrReadEn` 下降沿已经消失。

### 5.2 最终全量回归

最终 clean run 的关键结果如下：

```text
TB_COMPARE_OK beats_compared=12288 lanes_compared=98304 max_abs_err=1.1920929e-07 max_rel_err=3.39754005e-07
TB_UTIL cycles=18075 busy=17857 busy_pct=98.794 out_valid=12288 out_valid_pct=67.983 sram_write=12288 sram_write_pct=67.983 sram_read=12288 sram_read_pct=67.983 out_write_overlap=6696 out_write_overlap_pct=37.046
TB_BUSY_BREAKDOWN busy_nonout=5594 busy_nonout_pct=30.949 write_only=4824 read_only=0 write_read=768 idle=2 idle_pct=0.011
```

可以确认：

1. 数值误差维持在单精度可接受范围内。
2. DDR 读、SRAM 写、输出阶段能够稳定重叠。
3. token 边界的控制气泡被去除后，整体周期数进一步改善。

## 6. 本轮经验

### 6.1 不要手改生成 RTL

这轮问题最终都回到了源模型和控制器模板层面修复。经验很明确：

1. `DUTPacked.sv`、`CoreDUT.sv`、`Controller.sv` 只用于定位问题。
2. 真正修改必须落在 Simulink 建模脚本或模板文件。
3. 否则下一次 HDL 重新生成会把手改内容全部覆盖。

### 6.2 波形异常先证明，再解释

`ddrReadEn` 的下降沿一开始看起来像“token 正常切换”，但用户明确指出它不合理。事实也证明：

1. 不能只凭表面 gating 条件解释波形。
2. 必须把关键内部状态打印出来。
3. 只有看到 `requestBeatIndex`、`loadActive`、token 索引和 bank 状态的联动，才能把根因钉死。

### 6.3 流水线正确不等于边界正确

即便总体 compare 通过，也不代表控制逻辑已经最优或完全正确。边界类 bug 常见特点是：

1. 大部分周期行为正常。
2. 只有 token 切换、bank 切换、尾拍和首拍附近出问题。
3. 如果只看平均吞吐，很容易漏掉细粒度气泡和错位。

### 6.4 输出因子必须与输出 token 严格绑定

对这类 streaming 设计，任何跨 token 的共享控制量都要警惕：

1. 生成它的阶段和消费它的阶段可能跨越很多拍。
2. 只做 bank 级缓存不一定够。
3. 若消费窗口是 token 粒度，往往还需要输出侧再锁存一次。

## 7. 后续建议

1. 以后遇到吞吐异常，优先在 token 边界检查 `requestBeatIndex`、`receiveBeatIndex`、`outputBeatIndex` 的相位关系。
2. 对所有“按 token 生效”的控制量，优先考虑是否需要 output-side latch。
3. 保留一套窄窗口调试模板，专门打印 controller 内部状态，避免每次临时重写。
4. 在提交 HDL 改动时，把源码修复点和对应生成物一起提交，避免仓库状态不一致。
